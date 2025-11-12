import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Tuple, Dict

import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf

import envpool
import wandb

from purejaxql.utils.atari_wrapper import JaxLogEnvPoolWrapper


class CNN(nn.Module):
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        return x


class QNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        x = CNN(norm_type=self.norm_type)(x, train)
        x = nn.Dense(self.action_dim)(x)
        return x


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def eps_greedy_exploration(rng, q_vals, eps):
    """Select actions using epsilon-greedy policy."""
    rng_a, rng_e = jax.random.split(rng)
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    random_actions = jax.random.randint(
        rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
    )
    chosed_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape) < eps,
        random_actions,
        greedy_actions,
    )
    return chosed_actions


def create_step_env_fn(env, network, config, total_envs, jit=True):
    """
    Creates a jitted function for stepping through the environment.

    Note: For Atari, we may have extra test environments running in parallel.
    """

    def step_env(train_state, env_state, last_obs, rng, eps):

        def _single_step(carry, _):
            last_obs, env_state, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)

            # Get Q-values
            q_vals = network.apply(
                {"params": train_state.params, "batch_stats": train_state.batch_stats},
                last_obs,
                train=False,
            )

            # Select actions with epsilon-greedy
            _rngs = jax.random.split(rng_a, total_envs)
            eps_array = jnp.full(config["NUM_ENVS"], eps)

            # Add zero epsilon for test envs (greedy)
            if config.get("TEST_DURING_TRAINING", False):
                eps_array = jnp.concatenate([eps_array, jnp.zeros(config["TEST_ENVS"])])

            actions = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps_array)

            # Step environment (envpool handles all envs at once)
            new_obs, new_env_state, reward, done, info = env.step(env_state, actions)

            # Scale reward
            reward = config.get("REW_SCALE", 1) * reward

            # Store transition
            transition = {
                "obs": last_obs,
                "action": actions,
                "reward": reward,
                "done": done,
                "next_obs": new_obs,
                "q_val": q_vals,
            }

            return (new_obs, new_env_state, rng), (transition, info)

        # Collect NUM_STEPS transitions
        (new_obs, new_env_state, _), (transitions, infos) = jax.lax.scan(
            _single_step,
            (last_obs, env_state, rng),
            None,
            config["NUM_STEPS"],
        )

        return new_env_state, new_obs, transitions, infos

    if jit:
        return jax.jit(step_env)
    return step_env


def create_compute_targets_fn(network, config, jit=True):
    """Creates a jitted function for computing lambda-return targets."""

    def compute_targets(train_state, transitions):
        # Get Q-value for last next_obs
        last_q = network.apply(
            {"params": train_state.params, "batch_stats": train_state.batch_stats},
            transitions["next_obs"][-1],
            train=False,
        )
        last_q = jnp.max(last_q, axis=-1)

        def _get_target(lambda_returns_and_next_q, transition_slice):
            lambda_returns, next_q = lambda_returns_and_next_q

            # Bootstrap target
            target_bootstrap = (
                transition_slice["reward"]
                + config["GAMMA"] * (1 - transition_slice["done"]) * next_q
            )

            # Lambda return
            delta = lambda_returns - next_q
            lambda_returns = (
                target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
            )
            lambda_returns = (
                1 - transition_slice["done"]
            ) * lambda_returns + transition_slice["done"] * transition_slice["reward"]

            next_q = jnp.max(transition_slice["q_val"], axis=-1)
            return (lambda_returns, next_q), lambda_returns

        # Initialize for last step
        last_q = last_q * (1 - transitions["done"][-1])
        lambda_returns = transitions["reward"][-1] + config["GAMMA"] * last_q

        # Get transitions except last
        transitions_except_last = jax.tree_map(lambda x: x[:-1], transitions)

        # Compute targets in reverse
        _, targets = jax.lax.scan(
            _get_target,
            (lambda_returns, last_q),
            transitions_except_last,
            reverse=True,
        )

        # Concatenate with last return
        lambda_targets = jnp.concatenate([targets, lambda_returns[None, ...]])

        return lambda_targets

    if jit:
        return jax.jit(compute_targets)
    return compute_targets


def create_learn_fn(network, config, jit=True):
    """Creates a jitted function for the learning update on a minibatch."""

    def learn(train_state, obs_batch, action_batch, target_batch):

        def _loss_fn(params):
            q_vals, updates = network.apply(
                {"params": params, "batch_stats": train_state.batch_stats},
                obs_batch,
                train=True,
                mutable=["batch_stats"],
            )

            # Get Q-values for chosen actions
            chosen_action_qvals = jnp.take_along_axis(
                q_vals,
                jnp.expand_dims(action_batch, axis=-1),
                axis=-1,
            ).squeeze(axis=-1)

            # MSE loss
            loss = 0.5 * jnp.square(chosen_action_qvals - target_batch).mean()

            return loss, (updates, chosen_action_qvals)

        # Compute gradients
        (loss, (updates, qvals)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            train_state.params
        )

        # Apply gradients
        new_train_state = train_state.apply_gradients(grads=grads)
        new_train_state = new_train_state.replace(
            grad_steps=train_state.grad_steps + 1,
            batch_stats=updates["batch_stats"],
        )

        return new_train_state, loss, qvals.mean()

    if jit:
        return jax.jit(learn)
    return learn


def train(config, rng):

    # Setup
    print(f"Initializing training for {config['ENV_NAME']}...")

    # Create environment using envpool
    def make_env(num_envs):
        env = envpool.make(
            config["ENV_NAME"],
            env_type="gym",
            num_envs=num_envs,
            seed=config["SEED"],
            **config["ENV_KWARGS"],
        )
        env.num_envs = num_envs
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.name = config["ENV_NAME"]
        env = JaxLogEnvPoolWrapper(env)
        return env

    # Total envs includes test envs if testing during training
    total_envs = (
        (config["NUM_ENVS"] + config["TEST_ENVS"])
        if config.get("TEST_DURING_TRAINING", False)
        else config["NUM_ENVS"]
    )
    env = make_env(total_envs)

    # Calculate number of updates
    num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    num_updates_decay = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES"] = num_updates
    config["NUM_UPDATES_DECAY"] = num_updates_decay

    print(f"Total updates: {num_updates}")
    print(
        f"Steps per update: {config['NUM_STEPS']} x {config['NUM_ENVS']} = {config['NUM_STEPS'] * config['NUM_ENVS']}"
    )
    if config.get("TEST_DURING_TRAINING", False):
        print(f"Test environments: {config['TEST_ENVS']} (running in parallel)")

    # Create network (CNN for Atari)
    network = QNetwork(
        action_dim=env.single_action_space.n,
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
    )

    # Initialize network
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1, *env.single_observation_space.shape))
    network_variables = network.init(_rng, init_x, train=False)

    # Create optimizer
    lr_scheduler = optax.linear_schedule(
        init_value=config["LR"],
        end_value=1e-20,
        transition_steps=num_updates_decay
        * config["NUM_MINIBATCHES"]
        * config["NUM_EPOCHS"],
    )
    lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.radam(learning_rate=lr),
    )

    # Create training state
    train_state = CustomTrainState.create(
        apply_fn=network.apply,
        params=network_variables["params"],
        batch_stats=network_variables["batch_stats"],
        tx=tx,
    )

    # Create jitted functions
    jit = True  # you might disable jitting for easier debugging
    print("Creating jitted functions...")
    step_env_fn = create_step_env_fn(env, network, config, total_envs, jit=jit)
    compute_targets_fn = create_compute_targets_fn(network, config, jit=jit)
    learn_fn = create_learn_fn(network, config, jit=jit)

    # Epsilon scheduler
    eps_scheduler = optax.linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        config["EPS_DECAY"] * num_updates_decay,
    )

    # Initialize environments (envpool reset is NOT jitted)
    print("Initializing environments...")
    obs, env_state = env.reset()

    # Training loop
    print("Starting training loop...")
    print("=" * 80)

    start_time = time.time()

    for update in range(int(num_updates)):

        # 1. SAMPLE PHASE: Collect transitions
        rng, _rng = jax.random.split(rng)
        eps = eps_scheduler(train_state.n_updates)

        env_state, obs, transitions, infos = step_env_fn(
            train_state, env_state, obs, _rng, eps
        )

        # Split off test environment data if present
        if config.get("TEST_DURING_TRAINING", False):
            # Remove test envs from transitions (they're not used for training)
            transitions = jax.tree_map(
                lambda x: x[:, : -config["TEST_ENVS"]], transitions
            )

        # Update timestep counter (only training envs)
        train_state = train_state.replace(
            timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
        )

        # 2. COMPUTE TARGETS
        targets = compute_targets_fn(train_state, transitions)

        # 3. LEARNING PHASE: Update network
        batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
        minibatch_size = batch_size // config["NUM_MINIBATCHES"]

        all_losses = []
        all_qvals = []

        for epoch in range(int(config["NUM_EPOCHS"])):

            # Shuffle and create minibatches
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, batch_size)

            # Reshape transitions and targets for minibatches
            # (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)
            obs_batch = transitions["obs"].reshape(
                batch_size, *transitions["obs"].shape[2:]
            )
            action_batch = transitions["action"].reshape(batch_size)
            target_batch = targets.reshape(batch_size)  # targets are scalars

            # Permute
            obs_batch = obs_batch[permutation]
            action_batch = action_batch[permutation]
            target_batch = target_batch[permutation]

            # Process each minibatch
            for mb in range(int(config["NUM_MINIBATCHES"])):
                start_idx = mb * minibatch_size
                end_idx = start_idx + minibatch_size

                mb_obs = obs_batch[start_idx:end_idx]
                mb_actions = action_batch[start_idx:end_idx]
                mb_targets = target_batch[start_idx:end_idx]

                # Perform learning step
                train_state, loss, qvals = learn_fn(
                    train_state, mb_obs, mb_actions, mb_targets
                )

                all_losses.append(loss)
                all_qvals.append(qvals)

        # Update counter
        train_state = train_state.replace(n_updates=train_state.n_updates + 1)

        # 4. LOGGING
        # Calculate env_frame (number of frames seen)
        env_frame = (
            train_state.timesteps * env.observation_space.shape[0]
        )  # stacked frames

        metrics = {
            "env_step": int(train_state.timesteps),
            "update_steps": int(train_state.n_updates),
            "env_frame": int(env_frame),
            "grad_steps": int(train_state.grad_steps),
            "td_loss": float(np.mean(all_losses)),
            "qvals": float(np.mean(all_qvals)),
            "epsilon": float(eps),
        }

        # Add environment metrics
        # Split train and test metrics if testing during training
        if config.get("TEST_DURING_TRAINING", False):
            test_infos = jax.tree_map(lambda x: x[:, -config["TEST_ENVS"] :], infos)
            train_infos = jax.tree_map(lambda x: x[:, : -config["TEST_ENVS"]], infos)

            for k, v in train_infos.items():
                metrics[k] = float(jnp.mean(v))
            for k, v in test_infos.items():
                metrics[f"test/{k}"] = float(jnp.mean(v))
        else:
            for k, v in infos.items():
                metrics[k] = float(jnp.mean(v))

        # 5. WANDB LOGGING
        if config["WANDB_MODE"] != "disabled":
            wandb.log(metrics, step=metrics["update_steps"])

        # 6. CONSOLE OUTPUT
        if update % max(1, num_updates // 20) == 0:  # Print ~20 times during training
            elapsed = time.time() - start_time
            steps_per_sec = train_state.timesteps / elapsed
            print(
                f"Update {update}/{num_updates} | "
                f"Steps: {train_state.timesteps} | "
                f"Frames: {env_frame} | "
                f"Loss: {metrics['td_loss']:.4f} | "
                f"Q-vals: {metrics['qvals']:.2f} | "
                f"Eps: {eps:.3f} | "
                f"SPS: {steps_per_sec:.0f}"
            )

            if "returned_episode_returns" in metrics:
                print(
                    f"  → Train Episode Return: {metrics['returned_episode_returns']:.2f}"
                )
            if "test/returned_episode_returns" in metrics:
                print(
                    f"  → Test Episode Return: {metrics['test/returned_episode_returns']:.2f}"
                )

    print("=" * 80)
    print(f"Training complete! Total time: {time.time() - start_time:.2f}s")

    return train_state, metrics


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
    train_state, final_metrics = train(config, rng)

    # Save model
    if config.get("SAVE_PATH", None) is not None:
        from purejaxql.utils.save_load import save_params

        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save config
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        # Save params
        save_path = os.path.join(
            save_dir,
            f'{alg_name}_{env_name}_seed{config["SEED"]}_simple.safetensors',
        )
        save_params(train_state.params, save_path)
        print(f"Model saved to {save_path}")

    wandb.finish()

    return train_state, final_metrics


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))

    if config.get("HYP_TUNE", False):
        print("Hyperparameter tuning not supported in simple version.")
        print("Please use the original pqn_atari.py for tuning.")
    else:
        single_run(config)


if __name__ == "__main__":
    main()
