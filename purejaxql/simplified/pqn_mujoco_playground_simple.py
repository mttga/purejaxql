import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Tuple, Dict, Sequence, NamedTuple

import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf

import wandb

from purejaxql.utils.brax_wrappers import (
    LogVecWrapper,
    PlaygroundVecGymnaxWrapper,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)
from purejaxql.utils.save_load import save


class Actor(nn.Module):
    """Deterministic policy network that outputs continuous actions."""

    action_dim: Sequence[int]
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    hidden_sizes: Sequence[int]
    activation: str = "relu"
    norm_type: str = "layer_norm"
    norm_input: bool = False
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, x, train=False):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm(epsilon=1e-6)(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(
                use_running_average=not train, epsilon=1e-5
            )(x)
        else:
            normalize = lambda x: x

        for hs in self.hidden_sizes:
            x = nn.Dense(hs, kernel_init=nn.initializers.orthogonal(self.init_scale))(x)
            x = normalize(x)
            x = activation(x)

        x = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale))(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias

        return x


class Critic(nn.Module):
    """Q-function network that takes state and action as input."""

    hidden_sizes: Sequence[int]
    norm_type: str = "layer_norm"
    norm_input: bool = False
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, x, action, train=False):
        x = jnp.concatenate([x, action], axis=-1)

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm(epsilon=1e-6)(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(
                use_running_average=not train, epsilon=1e-5
            )(x)
        else:
            normalize = lambda x: x

        for hs in self.hidden_sizes:
            x = nn.Dense(hs, kernel_init=nn.initializers.orthogonal(self.init_scale))(x)
            x = normalize(x)
            x = nn.relu(x)

        x = nn.Dense(1, kernel_init=nn.initializers.orthogonal(self.init_scale))(x)

        return jnp.squeeze(x, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    original_action: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    noise: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def smooth_l1_loss(pred, target, beta=1.0):
    """Huber loss for robust Q-value regression."""
    diff = pred - target
    abs_diff = jnp.abs(diff)
    loss = jnp.where(
        abs_diff < beta,
        0.5 * (diff**2) / beta,
        abs_diff - 0.5 * beta,
    )
    return loss


def create_actor_critic_step_fn(actor, critic, env, config, jit=True):
    """Creates a jitted function for selecting actions and estimating values."""

    def actor_critic_step(train_state, obs, rng, noise_std):

        # Get action from actor
        action = actor.apply(
            {
                "params": train_state["actor"].params,
                "batch_stats": train_state["actor"].batch_stats,
            },
            obs["actor"],
            train=False,
        )

        original_action = action.copy()

        # Add exploration noise
        rng, _rng = jax.random.split(rng)
        if config.get("LINSPACE_NOISE", False):
            noise_stds = jnp.linspace(0, noise_std, config["NUM_ENVS"])
        else:
            noise_stds = jnp.full((config["NUM_ENVS"],), noise_std)

        noise = (
            jax.random.normal(_rng, action.shape)
            * noise_stds[:, np.newaxis]
            * actor.action_scale
        )
        action = action + noise
        action = env.clip_action(action)

        # Get Q-values from critics and average them
        def single_critic_step(critic_params, batch_stats):
            value = critic.apply(
                {"params": critic_params, "batch_stats": batch_stats},
                obs["critic"],
                action,
                train=False,
            )
            return value

        critic_train_state = train_state["critic"]
        values = jax.vmap(single_critic_step)(
            critic_train_state.params, critic_train_state.batch_stats
        )
        value = jnp.mean(values, axis=0)

        return original_action, action, value, noise

    if jit:
        return jax.jit(actor_critic_step)
    return actor_critic_step


def create_step_env_fn(env, env_params, actor_critic_step_fn, config, jit=True):
    """Creates a jitted function for collecting trajectories."""

    def step_env(train_state, env_state, last_obs, rng, noise_std):

        def _single_step(carry, _):
            train_state, env_state, last_obs, rng = carry

            # Select action and get value
            original_action, action, value, noise = actor_critic_step_fn(
                train_state, last_obs, rng, noise_std
            )

            # Step environment
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            new_obs, new_env_state, reward, done, info = env.step(
                rng_step, env_state, action, env_params
            )

            transition = Transition(
                done=done,
                original_action=original_action,
                action=action,
                value=value,
                reward=reward,
                noise=noise,
                obs=last_obs,
                next_obs=new_obs,
                info=info,
            )

            carry = (train_state, new_env_state, new_obs, rng)
            return carry, transition

        # Collect transitions
        (train_state, new_env_state, new_obs, rng), transitions = jax.lax.scan(
            _single_step,
            (train_state, env_state, last_obs, rng),
            None,
            config["NUM_STEPS"],
        )

        # Get action and value for final state (for bootstrap)
        rng, _rng = jax.random.split(rng)
        _, final_action, final_value, _ = actor_critic_step_fn(
            train_state, new_obs, _rng, noise_std
        )

        return new_env_state, new_obs, transitions, final_action, final_value

    if jit:
        return jax.jit(step_env)
    return step_env


def create_compute_targets_fn(config, jit=True):
    """Creates a jitted function for computing lambda-return targets."""

    def compute_targets(transitions, final_value):

        def _get_target(lambda_returns_and_next_q, transition_slice):
            lambda_returns, next_q = lambda_returns_and_next_q

            target_bootstrap = (
                transition_slice.reward
                + config["GAMMA"] * (1 - transition_slice.done) * next_q
            )
            delta = lambda_returns - next_q
            lambda_returns = (
                target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
            )
            lambda_returns = (
                1 - transition_slice.done
            ) * lambda_returns + transition_slice.done * transition_slice.reward
            next_q = transition_slice.value
            return (lambda_returns, next_q), lambda_returns

        # Initialize with final state
        final_value = final_value * (1 - transitions.done[-1])
        lambda_returns = transitions.reward[-1] + config["GAMMA"] * final_value

        # Get all transitions except last
        transitions_except_last = jax.tree_util.tree_map(lambda x: x[:-1], transitions)

        # Compute targets in reverse
        _, targets = jax.lax.scan(
            _get_target,
            (lambda_returns, final_value),
            transitions_except_last,
            reverse=True,
        )

        # Concatenate with final return
        targets = jnp.concatenate([targets, lambda_returns[None, ...]])

        return targets

    if jit:
        return jax.jit(compute_targets)
    return compute_targets


def create_learn_critic_fn(critic, config, jit=True):
    """Creates a jitted function for updating critics."""

    def learn_critic(train_state_critic, obs_batch, action_batch, target_batch):

        def _critic_loss_fn(critic_params, batch_stats):
            """Loss for a single critic in the ensemble."""

            def single_critic_pass(params, batch_stats_single):
                value, updates = critic.apply(
                    {"params": params, "batch_stats": batch_stats_single},
                    obs_batch,
                    action_batch,
                    train=True,
                    mutable=["batch_stats"],
                )
                return value, updates

            values, updates = jax.vmap(single_critic_pass)(critic_params, batch_stats)

            # Smooth L1 loss for each critic
            value_losses = jax.vmap(smooth_l1_loss, in_axes=(0, None))(
                values, target_batch
            )
            losses = jax.vmap(lambda x: jnp.mean(x))(value_losses)
            loss = jnp.sum(losses)

            loss_infos = {
                "value_loss": value_losses.mean(),
            }

            return loss, (updates, loss_infos)

        # Compute gradients
        (loss, (updates, loss_infos)), grads = jax.value_and_grad(
            _critic_loss_fn, has_aux=True
        )(train_state_critic.params, train_state_critic.batch_stats)

        # Apply gradients
        new_train_state = train_state_critic.apply_gradients(grads=grads)
        new_train_state = new_train_state.replace(
            grad_steps=train_state_critic.grad_steps + 1,
            batch_stats=updates["batch_stats"],
        )

        return new_train_state, loss, loss_infos

    if jit:
        return jax.jit(learn_critic)
    return learn_critic


def create_learn_actor_fn(actor, critic, config, jit=True):
    """Creates a jitted function for updating the actor."""

    def learn_actor(
        train_state, obs_actor_batch, obs_critic_batch, original_action_batch
    ):

        # Cache train state components outside gradient function
        actor_batch_stats = train_state["actor"].batch_stats
        critic_params = train_state["critic"].params
        critic_batch_stats = train_state["critic"].batch_stats

        def _actor_loss_fn(actor_params):
            # Get action from actor
            action, updates_actor = actor.apply(
                {
                    "params": actor_params,
                    "batch_stats": actor_batch_stats,
                },
                obs_actor_batch,
                train=True,
                mutable=["batch_stats"],
            )

            # Evaluate action with critics
            def single_critic_q(action, critic_p, batch_stats_c):
                q_val = critic.apply(
                    {"params": critic_p, "batch_stats": batch_stats_c},
                    obs_critic_batch,
                    action,
                    train=False,
                )
                return q_val

            q_values = jax.vmap(single_critic_q, in_axes=(None, 0, 0))(
                action, critic_params, critic_batch_stats
            )
            q_value = jnp.mean(q_values, axis=0)

            # RL loss (negative Q-value to maximize Q)
            rl_loss = jnp.mean(q_value, axis=0)

            # Penalty loss (to keep actions close to original noisy actions)
            action_diff = action - original_action_batch
            action_diff = (action_diff - actor.action_bias) / actor.action_scale
            action_diff = jnp.abs(action_diff).mean(axis=-1)
            pen_loss = smooth_l1_loss(action, original_action_batch).mean(axis=-1)

            penalty = jnp.where(
                action_diff < config["THRESHOLD"],
                0.0,
                config["PENALTY_COEFF"] * pen_loss,
            )

            # Actor loss is negative rl_loss plus penalty
            actor_loss = jnp.mean(-rl_loss + penalty)

            return actor_loss, updates_actor

        # Compute gradients
        (loss, updates_actor), grads = jax.value_and_grad(_actor_loss_fn, has_aux=True)(
            train_state["actor"].params
        )

        # Apply gradients
        new_train_state_actor = train_state["actor"].apply_gradients(grads=grads)
        new_train_state_actor = new_train_state_actor.replace(
            grad_steps=train_state["actor"].grad_steps + 1,
            batch_stats=updates_actor["batch_stats"],
        )

        return new_train_state_actor, loss

    if jit:
        return jax.jit(learn_actor)
    return learn_actor


def create_test_fn(actor, env, env_params, config, jit=True):
    """Creates a function for testing the trained agent."""

    def test_agent(train_state, training_env_state, rng):
        """Run test episodes and return metrics."""

        def _env_step(carry, _):
            env_state, last_obs, rng, returns = carry

            # Get action from actor (no noise during testing)
            action = actor.apply(
                {
                    "params": train_state["actor"].params,
                    "batch_stats": train_state["actor"].batch_stats,
                },
                last_obs["actor"],
                train=False,
            )

            # Step environment
            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, config["TEST_NUM_ENVS"])
            new_obs, new_env_state, reward, done, info = env.step(
                _rng, env_state, action, env_params
            )

            # Track returns only for running episodes
            returns["running_returns"] = jnp.where(
                ~returns["running_done"],
                returns["running_returns"] + info["original_reward"],
                returns["running_returns"],
            )
            returns["running_len"] = jnp.where(
                ~returns["running_done"],
                returns["running_len"] + 1,
                returns["running_len"],
            )
            returns["running_done"] = (returns["running_done"] + done).astype(bool)

            return (new_env_state, new_obs, rng, returns), info

        # Reset test environments
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["TEST_NUM_ENVS"])
        init_obs, reset_env_state = env.reset(reset_rng, env_params)

        # Apply observation normalization if enabled
        if config["NORMALIZE_OBS"]:
            env_state = training_env_state.replace(
                env_state=reset_env_state.env_state,
            )
            init_obs["actor"] = (init_obs["actor"] - env_state.actor_mean) / jnp.sqrt(
                env_state.actor_var + 1e-8
            )
        else:
            env_state = reset_env_state

        # Initialize return tracking
        returns = {
            "running_returns": jnp.zeros((config["TEST_NUM_ENVS"],)),
            "running_len": jnp.zeros((config["TEST_NUM_ENVS"],)),
            "running_done": jnp.zeros((config["TEST_NUM_ENVS"],), dtype=bool),
        }

        # Run test episodes
        (new_env_state, new_obs, rng, returns), infos = jax.lax.scan(
            _env_step,
            (env_state, init_obs, _rng, returns),
            None,
            config["TEST_NUM_STEPS"],
        )

        # Compute metrics
        test_metrics = {
            "test/returned_episode_returns": returns["running_returns"].sum()
            / jnp.maximum(returns["running_done"].sum(), 1),
            "test/returned_episode_lengths": returns["running_len"].sum()
            / jnp.maximum(returns["running_done"].sum(), 1),
            "test/done_episodes": returns["running_done"].sum()
            / config["TEST_NUM_ENVS"],
        }

        return test_metrics

    if jit:
        return jax.jit(test_agent)
    return test_agent


def train(config, rng):

    # Setup
    print(f"Initializing training for {config['ENV_NAME']}...")

    # Create environment with wrappers
    env, env_params = PlaygroundVecGymnaxWrapper(config["ENV_NAME"]), None
    print(f"Episode length: {env.episode_length}")

    action_space = env.action_space(None)
    env = LogVecWrapper(env)
    env = ClipAction(env, low=action_space.low, high=action_space.high)

    if config["NORMALIZE_REWARD"]:
        env = NormalizeVecReward(env, config["GAMMA"])
    if config["NORMALIZE_OBS"]:
        env = NormalizeVecObservation(env)

    # Calculate number of updates
    num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["NUM_UPDATES"] = num_updates
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Setup testing
    if config.get("TEST_DURING_TRAINING", True):
        config["TEST_NUM_STEPS"] = env.episode_length
        config["TEST_NUM_ENVS"] = (
            config["TEST_NUM_ENVS"]
            if config.get("TEST_NUM_ENVS") is not None
            else config["NUM_ENVS"]
        )
        print(f"Test num steps: {config['TEST_NUM_STEPS']}")
        print(f"Test num envs: {config['TEST_NUM_ENVS']}")

    print(f"Total updates: {num_updates}")
    print(
        f"Steps per update: {config['NUM_STEPS']} x {config['NUM_ENVS']} = {config['NUM_STEPS'] * config['NUM_ENVS']}"
    )

    # Create networks
    actor = Actor(
        action_dim=env.action_space(env_params).shape[0],
        action_scale=jnp.array((env.high - env.low) / 2.0),
        action_bias=jnp.array((env.high + env.low) / 2.0),
        hidden_sizes=config["ACTOR_HIDDEN_SIZES"],
        activation=config.get("ACTIVATION", "relu"),
        norm_type=config["NORM_TYPE"],
        init_scale=config.get("ACTOR_INIT_SCALE", 1.0),
    )

    critic = Critic(
        hidden_sizes=config["CRITIC_HIDDEN_SIZES"],
        norm_type=config["NORM_TYPE"],
        init_scale=config.get("CRITIC_INIT_SCALE", 1.0),
    )

    # Initialize networks
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params)["actor"].shape)
    actor_variables = actor.init(_rng, init_x)

    init_x = jnp.zeros(env.observation_space(env_params)["critic"].shape)
    dummy_action = jnp.zeros(env.action_size)
    rng, _rng = jax.random.split(rng)
    _rngs = jax.random.split(_rng, config["NUM_CRITICS"])
    critic_variables = jax.vmap(critic.init, in_axes=(0, None, None))(
        _rngs, init_x, dummy_action
    )

    # Create optimizers
    lr_scheduler = optax.linear_schedule(
        init_value=config["LR_START"],
        end_value=config["LR_END"],
        transition_steps=num_updates
        * config["LR_DECAY"]
        * config["NUM_MINIBATCHES"]
        * config["NUM_EPOCHS"],
    )
    lr = lr_scheduler if config.get("ANNEAL_LR", False) else config["LR_START"]

    tx_actor = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.radam(learning_rate=lr),
    )
    tx_critic = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.radam(learning_rate=lr),
    )

    # Create training states
    train_state_actor = CustomTrainState.create(
        apply_fn=actor.apply,
        params=actor_variables["params"],
        batch_stats=actor_variables["batch_stats"],
        tx=tx_actor,
    )
    train_state_critic = CustomTrainState.create(
        apply_fn=critic.apply,
        params=critic_variables["params"],
        batch_stats=critic_variables["batch_stats"],
        tx=tx_critic,
    )
    train_state = {"actor": train_state_actor, "critic": train_state_critic}

    # Create jitted functions
    jit = True
    print("Creating jitted functions...")
    actor_critic_step_fn = create_actor_critic_step_fn(
        actor, critic, env, config, jit=jit
    )
    step_env_fn = create_step_env_fn(
        env, env_params, actor_critic_step_fn, config, jit=jit
    )
    compute_targets_fn = create_compute_targets_fn(config, jit=jit)
    learn_critic_fn = create_learn_critic_fn(critic, config, jit=jit)
    learn_actor_fn = create_learn_actor_fn(actor, critic, config, jit=jit)
    test_fn = create_test_fn(actor, env, env_params, config, jit=jit)

    # Noise scheduler
    noise_scheduler = optax.linear_schedule(
        init_value=config["NOISE_START"],
        end_value=config["NOISE_FINISH"],
        transition_steps=config["NOISE_DECAY"] * num_updates,
    )

    # Initialize environments
    print("Initializing environments...")
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obs, env_state = env.reset(reset_rng, env_params)

    # Training loop
    print("Starting training loop...")
    print("=" * 80)

    start_time = time.time()

    for update in range(int(num_updates)):

        # 1. COLLECT TRAJECTORIES
        rng, _rng = jax.random.split(rng)
        noise_std = noise_scheduler(train_state["actor"].n_updates)

        env_state, obs, transitions, final_action, final_value = step_env_fn(
            train_state, env_state, obs, _rng, noise_std
        )

        # Update timestep counter
        train_state["actor"] = train_state["actor"].replace(
            timesteps=train_state["actor"].timesteps
            + config["NUM_STEPS"] * config["NUM_ENVS"]
        )

        # 2. COMPUTE TARGETS
        targets = compute_targets_fn(transitions, final_value)

        # 3. UPDATE NETWORKS
        batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
        minibatch_size = config["MINIBATCH_SIZE"]

        all_critic_losses = []
        all_actor_losses = []

        for epoch in range(int(config["NUM_EPOCHS"])):

            # Shuffle data
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, batch_size)

            # Reshape and permute
            obs_actor = transitions.obs["actor"].reshape(batch_size, -1)
            obs_critic = transitions.obs["critic"].reshape(batch_size, -1)
            actions = transitions.action.reshape(batch_size, -1)
            original_actions = transitions.original_action.reshape(batch_size, -1)
            targets_batch = targets.reshape(batch_size)

            obs_actor = obs_actor[permutation]
            obs_critic = obs_critic[permutation]
            actions = actions[permutation]
            original_actions = original_actions[permutation]
            targets_batch = targets_batch[permutation]

            # Process minibatches
            for mb in range(int(config["NUM_MINIBATCHES"])):
                start_idx = mb * minibatch_size
                end_idx = start_idx + minibatch_size

                mb_obs_actor = obs_actor[start_idx:end_idx]
                mb_obs_critic = obs_critic[start_idx:end_idx]
                mb_actions = actions[start_idx:end_idx]
                mb_original_actions = original_actions[start_idx:end_idx]
                mb_targets = targets_batch[start_idx:end_idx]

                # Update critic
                train_state["critic"], critic_loss, critic_loss_info = learn_critic_fn(
                    train_state["critic"], mb_obs_critic, mb_actions, mb_targets
                )
                all_critic_losses.append(critic_loss)

                # Update actor
                train_state["actor"], actor_loss = learn_actor_fn(
                    train_state, mb_obs_actor, mb_obs_critic, mb_original_actions
                )
                all_actor_losses.append(actor_loss)

        # Update counters
        train_state["actor"] = train_state["actor"].replace(
            n_updates=train_state["actor"].n_updates + 1
        )

        # 4. LOGGING
        metrics = {
            "env_step": int(train_state["actor"].timesteps),
            "update_steps": int(train_state["actor"].n_updates),
            "grad_steps_actor": int(train_state["actor"].grad_steps),
            "grad_steps_critic": int(train_state["critic"].grad_steps),
            "noise": float(noise_std),
            "loss": float(np.mean(all_critic_losses) + np.mean(all_actor_losses)),
            "value_loss": float(np.mean(all_critic_losses)),
            "loss_actor": float(np.mean(all_actor_losses)),
            "lr": (
                float(lr_scheduler(train_state["actor"].n_updates))
                if config.get("ANNEAL_LR", False)
                else config["LR_START"]
            ),
        }

        # Add environment metrics
        for k, v in transitions.info.items():
            metrics[k] = float(jnp.mean(v))

        # 5. RUN TESTS PERIODICALLY
        if config.get("TEST_DURING_TRAINING", False):
            test_interval = int(
                config["NUM_UPDATES"] * config.get("TEST_INTERVAL", 0.1)
            )
            if update % test_interval == 0:
                rng, _rng = jax.random.split(rng)
                test_metrics = test_fn(train_state, env_state, _rng)

                # Add test metrics
                for k, v in test_metrics.items():
                    metrics[k] = float(v)

        # 6. WANDB LOGGING
        if config["WANDB_MODE"] != "disabled":
            wandb.log(metrics, step=metrics["update_steps"])

        # 7. CONSOLE OUTPUT
        if update % max(1, num_updates // 20) == 0:
            elapsed = time.time() - start_time
            steps_per_sec = train_state["actor"].timesteps / elapsed
            print(
                f"Update {update}/{num_updates} | "
                f"Steps: {train_state['actor'].timesteps} | "
                f"Critic Loss: {metrics['value_loss']:.4f} | "
                f"Actor Loss: {metrics['loss_actor']:.4f} | "
                f"Noise: {noise_std:.3f} | "
                f"SPS: {steps_per_sec:.0f} | "
            )

            if "returned_episode_returns" in metrics:
                print(f"  → Episode Return: {metrics['returned_episode_returns']:.2f}")
            if "test/returned_episode_returns" in metrics:
                print(
                    f"  → Test Episode Return: {metrics['test/returned_episode_returns']:.2f}"
                )

    print("=" * 80)
    print(f"Training complete! Total time: {time.time() - start_time:.2f}s")

    return train_state, env_state, metrics


def single_run(config):
    """Run a single training run."""

    config = {**config, **config["alg"]}

    alg_name = f"{config.get('ALG_NAME', 'pqn')}_simple"
    env_name = config["ENV_NAME"]

    # Initialize wandb
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
            "simple_version",
        ],
        name=config.get("NAME", f"{alg_name}_{env_name}_simple"),
        config=config,
        mode=config["WANDB_MODE"],
    )

    # Run training
    rng = jax.random.PRNGKey(config["SEED"])
    train_state, env_state, final_metrics = train(config, rng)

    # Save model
    if config.get("SAVE_PATH", None) is not None:
        to_save = {
            "actor": {
                "params": train_state["actor"].params,
                "batch_stats": train_state["actor"].batch_stats,
            },
            "critic": {
                "params": train_state["critic"].params,
                "batch_stats": train_state["critic"].batch_stats,
            },
        }

        # Add normalization stats if they exist
        if config["NORMALIZE_OBS"] and hasattr(env_state, "actor_mean"):
            normalization_stats = {
                "actor_mean": env_state.actor_mean,
                "actor_var": env_state.actor_var,
                "critic_mean": env_state.critic_mean,
                "critic_var": env_state.critic_var,
            }
            to_save["normalization_stats"] = normalization_stats

        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        save_name = (
            f'{config["ALG_NAME"]}_{config["ENV_NAME"]}_seed{config["SEED"]}_simple'
        )
        save(to_save, config, save_dir, save_name, vmaps=1)
        print(f"Model saved to {save_dir}/{save_name}")

    wandb.finish()

    return train_state, final_metrics


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))

    if config.get("HYP_TUNE", False):
        print("Hyperparameter tuning not supported in simple version.")
        print("Please use the original pqn_mujoco_playground.py for tuning.")
    else:
        single_run(config)


if __name__ == "__main__":
    main()
