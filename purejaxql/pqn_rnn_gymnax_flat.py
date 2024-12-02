"""*Exactly* the same algorithm as `pqn_rnn_gymnax.py`, but with typing and reduced nesting."""

import copy
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generic

import chex
import flax.linen as nn
import flax.struct
import gymnax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from flax.typing import FrozenVariableDict
from gymnax.environments.environment import Environment, TEnvParams, TEnvState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from omegaconf import OmegaConf
from typing_extensions import NamedTuple, TypedDict
from xtils.jitpp import Static, jit
import wandb

# Reusing these from the `pqn_gymnax_flat` module.
from purejaxql.pqn_gymnax_flat import Config as BaseConfig
from purejaxql.pqn_gymnax_flat import save_params


# Extend the base config to add the entries that are specific to this RNN version.
class Config(BaseConfig, TypedDict):
    """TypedDict that gives type hints for the config dictionary."""

    # Note: This is not a structured config for Hydra, it's just here to help with type-checking the code.

    ENV_KWARGS: dict[str, Any]
    MEMORY_WINDOW: int  # steps of previous episode added in the rnn training horizon


class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: jax.Array, x: tuple[jax.Array, jax.Array]):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = rnn_state.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *resets.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size: int, *batch_size: int):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array):
        return x


class RNNQNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    norm_input: bool = False
    norm_type: str = "layer_norm"
    dueling: bool = False

    @nn.compact
    def __call__(self, hidden, x, done, last_action, train: bool = False):
        if self.norm_type == "layer_norm":
            normalize = nn.LayerNorm
        elif self.norm_type == "batch_norm":
            normalize = partial(nn.BatchNorm, use_running_average=not train)
        else:
            normalize = Identity

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input in any case for global compatibility
            # NOTE: This is probably done so there are always batch stats in the train state?
            _x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        for _layer_index in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize()(x)
            x = nn.relu(x)

        # add last action to the input of the rnn
        last_action = jax.nn.one_hot(last_action, self.action_dim)
        x = jnp.concatenate([x, last_action], axis=-1)

        rnn_in = (x, done)
        hidden, x = ScannedRNN()(hidden, rnn_in)

        q_vals = nn.Dense(self.action_dim)(x)

        return hidden, q_vals

    def initialize_carry(self, *batch_size: int):
        return ScannedRNN.initialize_carry(self.hidden_size, *batch_size)


class Transition(flax.struct.PyTreeNode):
    last_hs: jax.Array
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    last_done: jax.Array
    last_action: jax.Array
    q_vals: jax.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


class ExplorationState(NamedTuple, Generic[TEnvState]):
    hs: jax.Array
    obs: jax.Array
    done: jax.Array
    action: jax.Array
    env_state: TEnvState


class Results(TypedDict, Generic[TEnvState]):
    runner_state: tuple[
        CustomTrainState, Transition, ExplorationState[TEnvState], Any, chex.PRNGKey
    ]
    metrics: dict[str, jax.Array]


def _get_num_updates(config: Config):
    return config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]


def _get_num_updates_decay(config: Config):
    return config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]


def make_train(config: Config) -> Callable[[chex.PRNGKey], Results]:
    # config["NUM_UPDATES"] = (
    #     config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    # )

    # config["NUM_UPDATES_DECAY"] = (
    #     config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    # )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    env, env_params = gymnax.make(config["ENV_NAME"])
    if config["ENV_NAME"] == "MemoryChain-bsuite":
        from gymnax.environments.bsuite.memory_chain import EnvParams

        env_params = EnvParams(
            memory_length=config["ENV_KWARGS"].get("memory_length", 10)
        )
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    TEST_NUM_STEPS = config.get("TEST_NUM_STEPS", env_params.max_steps_in_episode)
    # config["TEST_NUM_STEPS"] = config.get(
    #     "TEST_NUM_STEPS", env_params.max_steps_in_episode
    # )
    return partial(
        train,
        config=FrozenDict(config),
        env=env,
        env_params=env_params,
        num_updates=_get_num_updates(config),
        num_updates_decay=_get_num_updates_decay(config),
        test_num_steps=TEST_NUM_STEPS,
    )


@jit
def train(
    rng: chex.PRNGKey,
    config: Static[dict],
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
    num_updates: Static[int],
    num_updates_decay: Static[int],
    test_num_steps: Static[int],
) -> Results[TEnvState]:
    original_rng = rng[0]

    eps_scheduler = optax.linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        (config["EPS_DECAY"]) * num_updates_decay,
    )
    lr_scheduler = optax.linear_schedule(
        init_value=config["LR"],
        end_value=1e-20,
        transition_steps=(num_updates_decay)
        * config["NUM_MINIBATCHES"]
        * config["NUM_EPOCHS"],
    )
    lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

    # INIT NETWORK AND OPTIMIZER
    network = RNNQNetwork(
        action_dim=env.action_space(env_params).n,
        hidden_size=config.get("HIDDEN_SIZE", 128),
        num_layers=config.get("NUM_LAYERS", 2),
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
    )

    rng, _rng = jax.random.split(rng)
    train_state = create_agent(
        rng,
        env=env,
        env_params=env_params,
        network=network,
        max_grad_norm=config["MAX_GRAD_NORM"],
        lr=lr,
    )

    # TRAINING LOOP
    rng, _rng = jax.random.split(rng)
    test_metrics: dict | None = get_test_metrics(
        train_state,
        _rng,
        config=config,
        network=network,
        env=env,
        env_params=env_params,
        test_num_steps=test_num_steps,
    )

    rng, _rng = jax.random.split(rng)
    obs, env_state = _vmap_reset(
        _rng, n_envs=config["NUM_ENVS"], env=env, env_params=env_params
    )
    init_dones = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
    init_action = jnp.zeros((config["NUM_ENVS"]), dtype=int)
    init_hs = network.initialize_carry(config["NUM_ENVS"])
    expl_state = ExplorationState(
        hs=init_hs, obs=obs, done=init_dones, action=init_action, env_state=env_state
    )

    # step randomly to have the initial memory window
    rng, _rng = jax.random.split(rng)
    (expl_state, rng), memory_transitions = jax.lax.scan(
        lambda exploration_state_and_rng, _: _random_step(
            exploration_state_and_rng,
            _,
            network=network,
            config=config,
            train_state=train_state,
            env=env,
            env_params=env_params,
        ),
        init=(expl_state, _rng),
        xs=None,
        length=config["MEMORY_WINDOW"] + config["NUM_STEPS"],
    )

    # train
    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, memory_transitions, expl_state, test_metrics, _rng)

    runner_state, metrics = jax.lax.scan(
        lambda runner_state_i, _update_step_index: _update_step(
            runner_state_i,
            _update_step_index,
            network=network,
            config=config,
            eps_scheduler=eps_scheduler,
            env=env,
            env_params=env_params,
            original_rng=original_rng,
            num_envs=config["NUM_ENVS"],
            num_steps=config["NUM_STEPS"],
            num_updates=num_updates,
            test_num_steps=test_num_steps,
        ),
        init=runner_state,
        xs=jnp.arange(num_updates),
        length=num_updates,
    )
    return {"runner_state": runner_state, "metrics": metrics}


@jit
def _random_step(
    carry: tuple[ExplorationState[TEnvState], chex.PRNGKey],
    _,
    network: Static[RNNQNetwork],
    train_state: CustomTrainState,
    config: Static[dict],
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
):
    (hs, last_obs, last_done, last_action, env_state), rng = carry
    rng, rng_a, rng_s = jax.random.split(rng, 3)
    assert isinstance(rng, chex.PRNGKey)
    _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
    _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
    _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)
    new_hs, q_vals = network.apply(
        {
            "params": train_state.params,
            "batch_stats": train_state.batch_stats,
        },
        hs,
        _obs,
        _done,
        _last_action,
        train=False,
    )  # (num_envs, hidden_size), (1, num_envs, num_actions)
    assert isinstance(q_vals, jax.Array)
    q_vals = q_vals.squeeze(axis=0)  # (num_envs, num_actions) remove the time dim
    _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
    eps = jnp.full(config["NUM_ENVS"], 1.0)  # random actions
    new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
    new_obs, new_env_state, reward, new_done, info = _vmap_step(
        rng_s,
        n_envs=config["NUM_ENVS"],
        env_state=env_state,
        action=new_action,
        env=env,
        env_params=env_params,
    )
    transition = Transition(
        last_hs=hs,
        obs=last_obs,
        action=new_action,
        reward=config.get("REW_SCALE", 1) * reward,
        done=new_done,
        last_done=last_done,
        last_action=last_action,
        q_vals=q_vals,
    )
    new_expl_state = ExplorationState(
        hs=new_hs,
        obs=new_obs,
        done=new_done,
        action=new_action,
        env_state=new_env_state,
    )
    return (new_expl_state, rng), transition


@jit
def get_test_metrics(
    train_state: CustomTrainState,
    rng: chex.PRNGKey,
    config: Static[dict],
    network: Static[RNNQNetwork],
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
    test_num_steps: Static[int],
):
    if not config.get("TEST_DURING_TRAINING", False):
        return None

    rng, _rng = jax.random.split(rng)
    init_obs, env_state = _vmap_reset(
        _rng, n_envs=config["TEST_NUM_ENVS"], env=env, env_params=env_params
    )
    init_done = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
    init_action = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=int)
    init_hs = network.initialize_carry(config["TEST_NUM_ENVS"])  # (n_envs, hs_size)
    step_state = (
        init_hs,
        init_obs,
        init_done,
        init_action,
        env_state,
        _rng,
    )
    step_state, infos = jax.lax.scan(
        lambda _step_state, _test_step_index: _greedy_env_step(
            _step_state,
            _test_step_index,
            env=env,
            env_params=env_params,
            network=network,
            train_state=train_state,
            config=config,
        ),
        init=step_state,
        xs=jnp.arange(test_num_steps),
        length=test_num_steps,
    )
    # return mean of done infos

    returned_episode: jax.Array = infos["returned_episode"]

    def _get_mean_of_done_episode_infos(x: jax.Array) -> jax.Array:
        return jnp.nanmean(jnp.where(returned_episode, x, jnp.nan))

    done_infos = jax.tree.map(_get_mean_of_done_episode_infos, infos)
    assert isinstance(done_infos, dict)
    return done_infos


@jit
def _greedy_env_step(
    step_state: tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, TEnvState, chex.PRNGKey
    ],
    _test_step_index: jax.Array,
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
    network: Static[RNNQNetwork],
    train_state: CustomTrainState,
    config: Static[dict],
):
    hs, last_obs, last_done, last_action, env_state, rng = step_state
    rng, rng_a, rng_s = jax.random.split(rng, 3)
    _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
    _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
    _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)
    new_hs, q_vals = network.apply(
        {
            "params": train_state.params,
            "batch_stats": train_state.batch_stats,
        },
        hs,
        _obs,
        _done,
        _last_action,
        train=False,
    )  # (num_envs, hidden_size), (1, num_envs, num_actions)
    assert isinstance(q_vals, jax.Array)
    q_vals = q_vals.squeeze(axis=0)  # (num_envs, num_actions) remove the time dim
    eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
    new_action = jax.vmap(eps_greedy_exploration)(
        jax.random.split(rng_a, config["TEST_NUM_ENVS"]), q_vals, eps
    )
    new_obs, new_env_state, reward, new_done, info = _vmap_step(
        rng_s,
        n_envs=config["TEST_NUM_ENVS"],
        env_state=env_state,
        action=new_action,
        env=env,
        env_params=env_params,
    )
    step_state = (new_hs, new_obs, new_done, new_action, new_env_state, rng)
    return step_state, info


@jit
def _update_step(
    runner_state: tuple[
        CustomTrainState,
        Transition,
        ExplorationState[TEnvState],
        dict | None,
        chex.PRNGKey,
    ],
    _update_step_index: jax.Array,
    network: Static[RNNQNetwork],
    config: Static[dict],
    eps_scheduler: Static[optax.Schedule],
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
    original_rng: chex.PRNGKey,
    num_envs: Static[int],
    num_steps: Static[int],
    test_num_steps: Static[int],
    num_updates: Static[int],
):
    (
        train_state,
        memory_transitions,
        expl_state,
        test_metrics,
        rng,
    ) = runner_state

    # SAMPLE PHASE
    # step the env
    rng, _rng = jax.random.split(rng)
    (expl_state, rng), (transitions, infos) = jax.lax.scan(
        lambda expl_state_and_rng, _step_index: _step_env(
            expl_state_and_rng,
            _step_index,
            network=network,
            train_state=train_state,
            eps_scheduler=eps_scheduler,
            env=env,
            env_params=env_params,
            num_envs=num_envs,
            reward_scaling_coef=config.get("REW_SCALE", 1),
        ),
        init=(expl_state, _rng),
        xs=jnp.arange(num_steps),
        length=num_steps,
    )

    train_state = train_state.replace(
        timesteps=train_state.timesteps + num_steps * num_envs
    )  # update timesteps count

    # insert the transitions into the memory
    memory_transitions: Transition = jax.tree.map(
        lambda x, y: jnp.concatenate([x[num_steps:], y], axis=0),
        memory_transitions,
        transitions,
    )

    # NETWORKS UPDATE
    rng, _rng = jax.random.split(rng)
    assert isinstance(rng, chex.PRNGKey)
    (train_state, rng), (loss, qvals) = jax.lax.scan(
        lambda train_state_and_rng, _epoch_index: _learn_epoch(
            train_state_and_rng,
            _epoch_index,  # epoch index (unused)
            config=config,
            network=network,
            memory_transitions=memory_transitions,
            num_minibatches=config["NUM_MINIBATCHES"],
        ),
        init=(train_state, rng),
        xs=jnp.arange(config["NUM_EPOCHS"]),
        length=config["NUM_EPOCHS"],
    )

    train_state = train_state.replace(n_updates=train_state.n_updates + 1)
    metrics = {
        "env_step": train_state.timesteps,
        "update_steps": train_state.n_updates,
        "grad_steps": train_state.grad_steps,
        "td_loss": loss.mean(),
        "qvals": qvals.mean(),
    }
    metrics.update({k: v.mean() for k, v in infos.items()})

    if config.get("TEST_DURING_TRAINING", False):
        rng, _rng = jax.random.split(rng)
        test_metrics = jax.lax.cond(
            train_state.n_updates % int(num_updates * config["TEST_INTERVAL"]) == 0,
            lambda _: get_test_metrics(
                train_state,
                _rng,
                config=config,
                network=network,
                env=env,
                env_params=env_params,
                test_num_steps=test_num_steps,
            ),
            lambda _: test_metrics,
            operand=None,
        )
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    # report on wandb if required
    if config["WANDB_MODE"] != "disabled":

        def callback(metrics: dict[str, Any], original_rng: jax.Array):
            if config.get("WANDB_LOG_ALL_SEEDS", False):
                metrics.update(
                    {f"rng{int(original_rng)}/{k}": v for k, v in metrics.items()}
                )
            wandb.log(metrics, step=metrics["update_steps"])

        jax.debug.callback(callback, metrics, original_rng)

    new_runner_state = (
        train_state,
        memory_transitions,
        expl_state,
        test_metrics,
        rng,
    )

    return new_runner_state, metrics


@jit
def _learn_epoch(
    carry: tuple[CustomTrainState, chex.PRNGKey],
    _epoch_index: jax.Array,
    config: Static[dict],
    network: Static[RNNQNetwork],
    memory_transitions: Transition,
    num_minibatches: Static[int],
):
    train_state, rng = carry
    rng, _rng = jax.random.split(rng)
    minibatches = jax.tree.map(
        partial(preprocess_transition, rng=_rng, num_minibatches=num_minibatches),
        memory_transitions,
    )  # num_minibatches, num_steps+memory_window, batch_size/num_minbatches, ...

    rng, _rng = jax.random.split(rng)
    (train_state, rng), (loss, qvals) = jax.lax.scan(
        lambda train_state_and_rng, minibatch: _learn_phase(
            train_state_and_rng, minibatch, config=config, network=network
        ),
        init=(train_state, rng),
        xs=minibatches,
    )

    return (train_state, rng), (loss, qvals)


@jit
def preprocess_transition(
    x: jax.Array, rng: chex.PRNGKey, num_minibatches: Static[int]
):
    # x: (num_steps, num_envs, ...)
    x = jax.random.permutation(rng, x, axis=1)  # shuffle the transitions
    x = x.reshape(
        x.shape[0], num_minibatches, -1, *x.shape[2:]
    )  # num_steps, minibatches, batch_size/num_minbatches,
    x = jnp.swapaxes(
        x, 0, 1
    )  # (minibatches, num_steps, batch_size/num_minbatches, ...)
    return x


@jit
def _learn_phase(
    train_state_and_rng: tuple[CustomTrainState, chex.PRNGKey],
    minibatch: Transition,
    config: Static[dict],
    network: Static[RNNQNetwork],
):
    # minibatch shape: num_steps, batch_size, ...
    # with batch_size = num_envs/num_minibatches

    train_state, rng = train_state_and_rng
    hs = minibatch.last_hs[0]  # hs of oldest step (batch_size, hidden_size)
    agent_in = (
        minibatch.obs,
        minibatch.last_done,
        minibatch.last_action,
    )
    loss_fn = partial(
        _loss_fn,
        network=network,
        train_state=train_state,
        hs=hs,
        agent_in=agent_in,
        minibatch=minibatch,
        gamma=config["GAMMA"],
        lambda_=config["LAMBDA"],
    )
    (loss, (updates, qvals)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        train_state.params
    )
    train_state = train_state.apply_gradients(grads=grads)
    train_state = train_state.replace(
        grad_steps=train_state.grad_steps + 1,
        batch_stats=updates["batch_stats"],
    )
    assert isinstance(loss, jax.Array)
    assert isinstance(qvals, jax.Array)
    return (train_state, rng), (loss, qvals)


@jit
def _loss_fn(
    params: FrozenVariableDict,
    network: Static[RNNQNetwork],
    train_state: CustomTrainState,
    hs: jax.Array,
    agent_in: tuple[jax.Array, ...],
    minibatch: Transition,
    gamma: float | jax.Array,
    lambda_: float | jax.Array,
):
    (_, q_vals), updates = network.apply(
        {"params": params, "batch_stats": train_state.batch_stats},
        hs,
        *agent_in,
        train=True,
        mutable=["batch_stats"],
    )  # (num_steps, batch_size, num_actions)

    # lambda returns are computed using NUM_STEPS as the horizon, and optimizing from t=0 to NUM_STEPS-1
    target_q_vals = jax.lax.stop_gradient(q_vals)
    last_q = target_q_vals[-1].max(axis=-1)
    target = _compute_targets(
        last_q,  # q_vals at t=NUM_STEPS-1
        target_q_vals[:-1],
        minibatch.reward[:-1],
        minibatch.done[:-1],
        gamma=gamma,
        lambda_=lambda_,
    ).reshape(-1)  # (num_steps-1*batch_size,)

    chosen_action_qvals = jnp.take_along_axis(
        q_vals,
        jnp.expand_dims(minibatch.action, axis=-1),
        axis=-1,
    ).squeeze(axis=-1)  # (num_steps, num_agents, batch_size,)
    chosen_action_qvals = chosen_action_qvals[:-1].reshape(
        -1
    )  # (num_steps-1*batch_size,)

    loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

    return loss, (updates, chosen_action_qvals)


@jit
def _compute_targets(
    last_q: jax.Array,
    q_vals: jax.Array,
    reward: jax.Array,
    done: jax.Array,
    gamma: float | jax.Array,
    lambda_: float | jax.Array,
):
    lambda_returns = reward[-1] + gamma * (1 - done[-1]) * last_q
    last_q = jnp.max(q_vals[-1], axis=-1)
    _, targets = jax.lax.scan(
        lambda carry, input: _get_target(
            carry,
            input,
            gamma=gamma,
            lambda_=lambda_,
        ),
        init=(lambda_returns, last_q),
        xs=jax.tree.map(lambda x: x[:-1], (reward, q_vals, done)),
        reverse=True,
    )
    targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
    return targets


def _get_target(
    lambda_returns_and_next_q: tuple[jax.Array, jax.Array],
    rew_q_done: tuple[jax.Array, jax.Array, jax.Array],
    gamma: float | jax.Array,
    lambda_: float | jax.Array,
):
    reward, q, done = rew_q_done
    lambda_returns, next_q = lambda_returns_and_next_q
    target_bootstrap = reward + gamma * (1 - done) * next_q
    delta = lambda_returns - next_q
    lambda_returns = target_bootstrap + gamma * lambda_ * delta
    lambda_returns = (1 - done) * lambda_returns + done * reward
    next_q = jnp.max(q, axis=-1)
    return (lambda_returns, next_q), lambda_returns


@jit
def _step_env(
    carry: tuple[ExplorationState[TEnvState], chex.PRNGKey],
    _,
    network: Static[RNNQNetwork],
    train_state: CustomTrainState,
    eps_scheduler: Static[optax.Schedule],
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
    num_envs: Static[int],
    reward_scaling_coef: float | jax.Array,
):
    (hs, last_obs, last_done, last_action, env_state), rng = carry
    rng, rng_a, rng_s = jax.random.split(rng, 3)

    _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
    _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
    _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)

    new_hs, q_vals = network.apply(
        {
            "params": train_state.params,
            "batch_stats": train_state.batch_stats,
        },
        hs,
        _obs,
        _done,
        _last_action,
        train=False,
    )  # (num_envs, hidden_size), (1, num_envs, num_actions)
    assert isinstance(q_vals, jax.Array)
    q_vals = q_vals.squeeze(axis=0)  # (num_envs, num_actions) remove the time dim

    _rngs = jax.random.split(rng_a, num_envs)
    eps = jnp.full(num_envs, eps_scheduler(train_state.n_updates))
    new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

    new_obs, new_env_state, reward, new_done, info = _vmap_step(
        rng_s,
        n_envs=num_envs,
        env=env,
        env_params=env_params,
        env_state=env_state,
        action=new_action,
    )

    transition = Transition(
        last_hs=hs,
        obs=last_obs,
        action=new_action,
        reward=reward_scaling_coef * reward,
        done=new_done,
        last_done=last_done,
        last_action=last_action,
        q_vals=q_vals,
    )
    new_expl_state = ExplorationState(
        hs=new_hs,
        obs=new_obs,
        done=new_done,
        action=new_action,
        env_state=new_env_state,
    )
    return (new_expl_state, rng), (
        transition,
        info,
    )


@jit
def create_agent(
    rng: chex.PRNGKey,
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
    network: Static[RNNQNetwork],
    max_grad_norm: Static[float],
    lr: Static[optax.Schedule | float],
) -> CustomTrainState:
    init_x = (
        jnp.zeros(
            (1, 1, *env.observation_space(env_params).shape)
        ),  # (time_step, batch_size, obs_size)
        jnp.zeros((1, 1)),  # (time_step, batch size)
        jnp.zeros((1, 1)),  # (time_step, batch size)
    )  # (obs, dones, last_actions)
    init_hs = network.initialize_carry(1)  # (batch_size, hidden_dim)
    network_variables = network.init(rng, init_hs, *init_x, train=False)
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.radam(learning_rate=lr),
    )

    train_state = CustomTrainState.create(
        apply_fn=network.apply,
        params=network_variables["params"],
        batch_stats=network_variables["batch_stats"],
        tx=tx,
    )
    return train_state


# epsilon-greedy exploration
def eps_greedy_exploration(
    rng: chex.PRNGKey, q_vals: jax.Array, eps: float | jax.Array
) -> jax.Array:
    rng_a, rng_e = jax.random.split(
        rng
    )  # a key for sampling random actions and one for picking
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    chosed_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape)
        < eps,  # pick the actions that should be random
        jax.random.randint(
            rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
        ),  # sample random actions,
        greedy_actions,
    )
    return chosed_actions


@jit
def _vmap_reset(
    rng: chex.PRNGKey,
    n_envs: Static[int],
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
) -> tuple[jax.Array, TEnvState]:
    return jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )


@jit
def _vmap_step(
    rng: chex.PRNGKey,
    env_state: gymnax.EnvState,
    action: jax.Array,
    n_envs: Static[int],
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
) -> tuple[jax.Array, TEnvState, jax.Array, jax.Array, dict]:
    return jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(rng, n_envs), env_state, action, env_params
    )


def single_run(_config: dict):
    config: Config = {**_config, **_config["alg"]}  # type: ignore

    alg_name = config.get("ALG_NAME", "pqn_rnn")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        # Added: Can use the `NAME` key in the config to override the default name for the run.
        name=config.get("NAME", f'{config["ALG_NAME"]}_{config["ENV_NAME"]}'),
        config=dict(config),
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"Took {time.time()-t0} seconds to complete.")

    if (save_path := config.get("SAVE_PATH")) is not None:
        model_state = outs["runner_state"][0]
        save_dir = Path(save_path) / env_name
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(_default_config):
    """Hyperparameter sweep with wandb."""

    default_config: Config = {**_default_config, **_default_config["alg"]}  # type: ignore
    alg_name = default_config.get("ALG_NAME", "pqn_rnn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.001,
                    0.0005,
                    0.0001,
                    0.00005,
                ]
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    assert isinstance(config, dict)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
