import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from mujoco_playground import registry
from mujoco_playground._src.wrapper import Wrapper as PlaygroundWrapper
from mujoco_playground._src.wrapper import wrap_for_brax_training


def get_original_state(state):
    if hasattr(state, "env_state"):
        return get_original_state(state.env_state)
    return state


def nan_warning(x, name="value"):
    if np.isnan(x).any():
        print(f"Warning: {name} contains nan values")
    if np.isinf(x).any():
        print(f"Warning: {name} contains inf values")


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class PlaygroundVecGymnaxWrapper(GymnaxWrapper):
    def __init__(self, env_name, custom_command=None):
        env_config = registry.get_default_config(env_name)
        env = registry.load(env_name, env_config)
        self.env_config = env_config
        self.action_scale = 1.0  # self.env_config.action_scale
        self.episode_length = self.env_config.episode_length
        self.action_repeat = self.env_config.action_repeat
        env = wrap_for_brax_training(
            env, episode_length=self.episode_length, action_repeat=self.action_repeat
        )
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        if isinstance(self._env.observation_size, dict):
            self.privileged_state = True
            print("Env has privileged state.")
        else:
            self.privileged_state = False
        self.custom_command = custom_command  # used for locomotion rendering

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        # inject custom command if provided (for rendering purposes)
        if self.custom_command is not None:
            state.info["command"] = state.info["command"].at[:].set(self.custom_command)

        if self.privileged_state:
            obs = {"actor": state.obs["state"], "critic": state.obs["privileged_state"]}
        else:
            obs = {"actor": state.obs, "critic": state.obs}
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        if self.privileged_state:
            obs = {
                "actor": next_state.obs["state"],
                "critic": next_state.obs["privileged_state"],
            }
        else:
            obs = {"actor": next_state.obs, "critic": next_state.obs}
        reward = next_state.reward
        jax.debug.callback(nan_warning, reward, name="reward")
        reward = jnp.where(
            jnp.isnan(reward), 0.0, reward
        )  # some envs might produce NaN rewards
        done = next_state.done > 0.5
        infos = {}
        return obs, next_state, reward, done, infos

    def observation_space(self, params):

        if self.privileged_state:
            return {
                "actor": spaces.Box(
                    low=-jnp.inf,
                    high=jnp.inf,
                    shape=self._env.observation_size["state"],
                ),
                "critic": spaces.Box(
                    low=-jnp.inf,
                    high=jnp.inf,
                    shape=self._env.observation_size["privileged_state"],
                ),
            }
        else:
            space = spaces.Box(
                low=-jnp.inf,
                high=jnp.inf,
                shape=(self._env.observation_size,),
            )
            return {"actor": space, "critic": space}

    def action_space(self, params):
        return spaces.Box(
            low=-self.action_scale,
            high=self.action_scale,
            shape=(self._env.action_size,),
        )


class NavixGymnaxWrapper:
    def __init__(self, env_name):
        self._env = nx.make(env_name)

    def reset(self, key, params=None):
        timestep = self._env.reset(key)
        return timestep.observation, timestep

    def step(self, key, state, action, params=None):
        timestep = self._env.step(state, action)
        return timestep.observation, timestep, timestep.reward, timestep.is_done(), {}

    def observation_space(self, params):
        return spaces.Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=(np.prod(self._env.observation_space.shape),),
            dtype=self._env.observation_space.dtype,
        )

    def action_space(self, params):
        return spaces.Discrete(
            num_categories=self._env.action_space.maximum.item() + 1,
        )


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = self.clip_action(action)
        return self._env.step(key, state, action, params)

    @partial(jax.jit, static_argnums=(0,))
    def clip_action(self, action):
        return jnp.clip(action, self.low, self.high)


class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    actor_mean: jnp.ndarray
    actor_var: jnp.ndarray
    actor_count: float
    critic_mean: jnp.ndarray
    critic_var: jnp.ndarray
    critic_count: float
    env_state: environment.EnvState


class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        # Initialize statistics for both actor and critic observations
        actor_obs = obs["actor"]
        critic_obs = obs["critic"]

        batch_size = actor_obs.shape[0]
        actor_mean = jnp.zeros_like(actor_obs[0])
        actor_var = jnp.ones_like(actor_obs[0])
        actor_count = 1e-4
        critic_mean = jnp.zeros_like(critic_obs[0])
        critic_var = jnp.ones_like(critic_obs[0])
        critic_count = 1e-4

        # Update statistics with initial batch
        actor_batch_mean = jnp.mean(actor_obs, axis=0)
        actor_batch_var = jnp.var(actor_obs, axis=0)
        actor_batch_count = batch_size

        delta_actor = actor_batch_mean - actor_mean
        actor_tot_count = actor_count + actor_batch_count
        new_actor_mean = actor_mean + delta_actor * actor_batch_count / actor_tot_count
        m_a_actor = actor_var * actor_count
        m_b_actor = actor_batch_var * actor_batch_count
        M2_actor = (
            m_a_actor
            + m_b_actor
            + jnp.square(delta_actor)
            * actor_count
            * actor_batch_count
            / actor_tot_count
        )
        new_actor_var = M2_actor / actor_tot_count
        new_actor_count = actor_tot_count

        critic_batch_mean = jnp.mean(critic_obs, axis=0)
        critic_batch_var = jnp.var(critic_obs, axis=0)
        critic_batch_count = batch_size

        delta_critic = critic_batch_mean - critic_mean
        critic_tot_count = critic_count + critic_batch_count
        new_critic_mean = (
            critic_mean + delta_critic * critic_batch_count / critic_tot_count
        )
        m_a_critic = critic_var * critic_count
        m_b_critic = critic_batch_var * critic_batch_count
        M2_critic = (
            m_a_critic
            + m_b_critic
            + jnp.square(delta_critic)
            * critic_count
            * critic_batch_count
            / critic_tot_count
        )
        new_critic_var = M2_critic / critic_tot_count
        new_critic_count = critic_tot_count

        state = NormalizeVecObsEnvState(
            actor_mean=new_actor_mean,
            actor_var=new_actor_var,
            actor_count=new_actor_count,
            critic_mean=new_critic_mean,
            critic_var=new_critic_var,
            critic_count=new_critic_count,
            env_state=state,
        )

        # Normalize observations
        norm_obs = {
            "actor": (actor_obs - state.actor_mean) / jnp.sqrt(state.actor_var + 1e-8),
            "critic": (critic_obs - state.critic_mean)
            / jnp.sqrt(state.critic_var + 1e-8),
        }
        return norm_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        actor_obs = obs["actor"]
        critic_obs = obs["critic"]
        batch_size = actor_obs.shape[0]

        # Update actor statistics
        actor_batch_mean = jnp.mean(actor_obs, axis=0)
        actor_batch_var = jnp.var(actor_obs, axis=0)
        actor_batch_count = batch_size

        delta_actor = actor_batch_mean - state.actor_mean
        actor_tot_count = state.actor_count + actor_batch_count
        new_actor_mean = (
            state.actor_mean + delta_actor * actor_batch_count / actor_tot_count
        )
        m_a_actor = state.actor_var * state.actor_count
        m_b_actor = actor_batch_var * actor_batch_count
        M2_actor = (
            m_a_actor
            + m_b_actor
            + jnp.square(delta_actor)
            * state.actor_count
            * actor_batch_count
            / actor_tot_count
        )
        new_actor_var = M2_actor / actor_tot_count
        new_actor_count = actor_tot_count

        # Update critic statistics
        critic_batch_mean = jnp.mean(critic_obs, axis=0)
        critic_batch_var = jnp.var(critic_obs, axis=0)
        critic_batch_count = batch_size

        delta_critic = critic_batch_mean - state.critic_mean
        critic_tot_count = state.critic_count + critic_batch_count
        new_critic_mean = (
            state.critic_mean + delta_critic * critic_batch_count / critic_tot_count
        )
        m_a_critic = state.critic_var * state.critic_count
        m_b_critic = critic_batch_var * critic_batch_count
        M2_critic = (
            m_a_critic
            + m_b_critic
            + jnp.square(delta_critic)
            * state.critic_count
            * critic_batch_count
            / critic_tot_count
        )
        new_critic_var = M2_critic / critic_tot_count
        new_critic_count = critic_tot_count

        state = NormalizeVecObsEnvState(
            actor_mean=new_actor_mean,
            actor_var=new_actor_var,
            actor_count=new_actor_count,
            critic_mean=new_critic_mean,
            critic_var=new_critic_var,
            critic_count=new_critic_count,
            env_state=env_state,
        )

        # Normalize observations
        norm_obs = {
            "actor": (actor_obs - state.actor_mean) / jnp.sqrt(state.actor_var + 1e-8),
            "critic": (critic_obs - state.critic_mean)
            / jnp.sqrt(state.critic_var + 1e-8),
        }
        return norm_obs, state, reward, done, info

    def step_without_update(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        # Normalize without updating statistics
        norm_obs = {
            "actor": (obs["actor"] - state.actor_mean)
            / jnp.sqrt(state.actor_var + 1e-8),
            "critic": (obs["critic"] - state.critic_mean)
            / jnp.sqrt(state.critic_var + 1e-8),
        }
        state = NormalizeVecObsEnvState(
            actor_mean=state.actor_mean,
            actor_var=state.actor_var,
            actor_count=state.actor_count,
            critic_mean=state.critic_mean,
            critic_var=state.critic_var,
            critic_count=state.critic_count,
            env_state=env_state,
        )
        return norm_obs, state, reward, done, info


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs["actor"].shape[0] if isinstance(obs, dict) else obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs["actor"].shape[0] if isinstance(obs, dict) else obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


@struct.dataclass
class LogVecEnvState:
    env_state: environment.EnvState
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    timestep: jnp.ndarray


class LogVecWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        batch_size = obs["actor"].shape[0] if isinstance(obs, dict) else obs.shape[0]
        state = LogVecEnvState(
            env_state,
            jnp.zeros(batch_size),
            jnp.zeros(batch_size),
            jnp.zeros(batch_size),
            jnp.zeros(batch_size),
            jnp.zeros(batch_size),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogVecEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        info["original_reward"] = reward
        return obs, state, reward, done, info
