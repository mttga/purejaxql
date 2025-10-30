import os
import time
import copy
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import wandb
import hydra
from omegaconf import OmegaConf
from purejaxql.utils.brax_wrappers import (
    LogVecWrapper,
    PlaygroundVecGymnaxWrapper,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)
from purejaxql.utils.save_load import save


class Actor(nn.Module):
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
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
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

    hidden_sizes: Sequence[int]
    norm_type: str = "layer_norm"
    norm_input: bool = False
    use_simba: bool = False
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, x, action, train=False):

        x = jnp.concatenate([x, action], axis=-1)

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
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
    next_action: jnp.ndarray
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


# huber loss
def smooth_l1_loss(pred, target, beta=1.0):
    diff = pred - target
    abs_diff = jnp.abs(diff)
    loss = jnp.where(
        abs_diff < beta,
        0.5 * (diff**2) / beta,
        abs_diff - 0.5 * beta,
    )
    return loss


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = PlaygroundVecGymnaxWrapper(config["ENV_NAME"]), None
    print("episode_length:", env.episode_length)
    action_space = env.action_space(None)
    env = LogVecWrapper(env)
    env = ClipAction(
        env,
        low=action_space.low,
        high=action_space.high,
    )
    if config["NORMALIZE_REWARD"]:
        env = NormalizeVecReward(env, config["GAMMA"])
    if config["NORMALIZE_OBS"]:
        env = NormalizeVecObservation(env)

    if config.get("TEST_DURING_TRAINING", True):
        config["TEST_NUM_STEPS"] = env.episode_length
        config["TEST_NUM_ENVS"] = config["NUM_ENVS"]
        print("Test num steps:", config["TEST_NUM_STEPS"])
        print("Test num envs:", config["TEST_NUM_ENVS"])

    lr_scheduler = optax.linear_schedule(
        init_value=config["LR_START"],
        end_value=config["LR_END"],
        transition_steps=(config["NUM_UPDATES"] * config["LR_DECAY"])
        * config["NUM_MINIBATCHES"]
        * config["NUM_EPOCHS"],
    )
    lr = lr_scheduler if config.get("ANNEAL_LR", False) else config["LR_START"]

    noise_scheduler = optax.linear_schedule(
        init_value=config["NOISE_START"],
        end_value=config["NOISE_FINISH"],
        transition_steps=(config["NOISE_DECAY"]) * config["NUM_UPDATES"],
    )
    log_times = []

    def train(rng):

        original_rng = rng[0]

        # INIT ACTOR
        actor = Actor(
            env.action_space(env_params).shape[0],
            action_scale=jnp.array((env.high - env.low) / 2.0),
            action_bias=jnp.array((env.high + env.low) / 2.0),
            hidden_sizes=config["ACTOR_HIDDEN_SIZES"],
            activation=config.get("ACTIVATION", "relu"),
            norm_type=config["NORM_TYPE"],
            init_scale=config.get("ACTOR_INIT_SCALE", 1.0),
        )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params)["actor"].shape)
        actor_variables = actor.init(_rng, init_x)

        # INIT CRITIC
        critic = Critic(
            hidden_sizes=config["CRITIC_HIDDEN_SIZES"],
            norm_type=config["NORM_TYPE"],
            init_scale=config.get("CRITIC_INIT_SCALE", 1.0),
        )
        init_x = jnp.zeros(env.observation_space(env_params)["critic"].shape)
        dummy_action = jnp.zeros(env.action_size)
        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, config["NUM_CRITICS"])
        critic_variables = jax.vmap(critic.init, in_axes=(0, None, None))(
            _rngs, init_x, dummy_action
        )

        tx_actor = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )
        tx_critic = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )

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

        def actor_critic_step(train_state, obs, rng, noise_std=0.0):
            # SELECT ACTION
            action = actor.apply(
                {
                    "params": train_state["actor"].params,
                    "batch_stats": train_state["actor"].batch_stats,
                },
                obs["actor"],
                train=False,
            )

            # add noise
            original_action = action.copy()
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

            def single_critic_step(critic_params, batch_stats):
                value = critic.apply(
                    {
                        "params": critic_params,
                        "batch_stats": batch_stats,
                    },
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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, test_metrics = runner_state

                noise_std = noise_scheduler(train_state["actor"].n_updates)
                original_action, action, value, noise = actor_critic_step(
                    train_state, last_obs, rng, noise_std
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                reward = reward
                transition = Transition(
                    done,
                    original_action,
                    action,
                    action,
                    value,
                    reward,
                    noise,
                    last_obs,
                    obsv,
                    info,
                )
                runner_state = (train_state, env_state, obsv, rng, test_metrics)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE Q(λ) targets
            train_state, env_state, last_obs, rng, test_metrics = runner_state
            rng, _rng = jax.random.split(rng)
            noise_std = noise_scheduler(train_state["actor"].n_updates)
            original_last_action, last_action, last_val, last_noise = actor_critic_step(
                train_state, last_obs, _rng, noise_std
            )
            next_actions = jnp.concatenate(
                (traj_batch.next_action[1:], last_action[np.newaxis])
            )
            traj_batch = traj_batch._replace(next_action=next_actions)

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                )
                lambda_returns = (
                    1 - transition.done
                ) * lambda_returns + transition.done * transition.reward
                next_q = transition.value
                return (lambda_returns, next_q), lambda_returns

            last_val = last_val * (1 - traj_batch.done[-1])
            lambda_returns = traj_batch.reward[-1] + config["GAMMA"] * last_val
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_val),
                jax.tree_util.tree_map(lambda x: x[:-1], traj_batch),
                reverse=True,
            )
            targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, targets = batch_info

                    def _critic_loss_fn(critic_params, traj_batch, targets):

                        # CALCULATE VALUE LOSS
                        if config.get("USE_QLAMBDA", True):

                            def single_critic_pass(params, batch_stats):
                                value, updates = critic.apply(
                                    {
                                        "params": params,
                                        "batch_stats": batch_stats,
                                    },
                                    traj_batch.obs["critic"],
                                    traj_batch.action,
                                    train=True,
                                    mutable=["batch_stats"],
                                )
                                return value, updates

                            values, updates = jax.vmap(single_critic_pass)(
                                critic_params, train_state["critic"].batch_stats
                            )

                        else:

                            def single_critic_pass(params, batch_stats):
                                all_obs = jnp.concatenate(
                                    (
                                        traj_batch.obs["critic"],
                                        traj_batch.next_obs["critic"],
                                    )
                                )
                                all_actions = jnp.concatenate(
                                    (
                                        traj_batch.action,
                                        traj_batch.next_action,
                                    )
                                )
                                all_q_vals, updates = critic.apply(
                                    {
                                        "params": params,
                                        "batch_stats": batch_stats,
                                    },
                                    all_obs,
                                    all_actions,
                                    train=True,
                                    mutable=["batch_stats"],
                                )
                                q_vals, q_next = jnp.split(all_q_vals, 2)
                                return q_vals, q_next, updates

                            values, next_values, updates = jax.vmap(single_critic_pass)(
                                critic_params, train_state["critic"].batch_stats
                            )
                            q_next = jnp.mean(next_values, axis=0)
                            targets = (
                                traj_batch.reward
                                + (1 - traj_batch.done) * config["GAMMA"] * q_next
                            )

                        value_diff = values - traj_batch.value
                        value_diff = jnp.abs(value_diff)

                        value_losses = jax.vmap(smooth_l1_loss, in_axes=(0, None))(
                            values, targets
                        )

                        losses = jax.vmap(lambda x: jnp.mean(x))(value_losses)
                        loss = jnp.sum(losses)

                        loss_infos = {
                            "value_loss": value_losses.mean(),
                            "critic_value_diff": value_diff.mean(),
                        }

                        return loss, (updates, loss_infos)

                    def _actor_loss_fn(actor_params, traj_batch):

                        # CALCULATE ACTOR LOSS
                        action, updates = actor.apply(
                            {
                                "params": actor_params,
                                "batch_stats": train_state["actor"].batch_stats,
                            },
                            traj_batch.obs["actor"],
                            train=True,
                            mutable=["batch_stats"],
                        )

                        def single_critic_value(action, params, batch_stats):
                            value = critic.apply(
                                {
                                    "params": params,
                                    "batch_stats": batch_stats,
                                },
                                traj_batch.obs["critic"],
                                action,
                                train=False,
                            )
                            return value

                        values = jax.vmap(single_critic_value, in_axes=(None, 0, 0))(
                            action,
                            train_state["critic"].params,
                            train_state["critic"].batch_stats,
                        )
                        rl_loss = jnp.mean(values, axis=0)  # values[0]

                        action_diff = action - traj_batch.original_action
                        action_diff = (
                            action_diff - actor.action_bias
                        ) / actor.action_scale
                        action_diff = jnp.abs(action_diff).mean(
                            axis=-1
                        )  # absolute difference
                        pen_loss = smooth_l1_loss(
                            action, traj_batch.original_action
                        ).mean(axis=-1)

                        penalty = jnp.where(
                            action_diff < config["THRESHOLD"],
                            0.0,
                            config["PENALTY_COEFF"] * pen_loss,
                        )
                        actor_loss = jnp.mean(-rl_loss + penalty)  # Scalar loss

                        loss_infos = {
                            "policy_loss": rl_loss.mean(),
                            "actor_penalty_loss": pen_loss.mean(),
                            "actor_penalty_ratio": (
                                action_diff > config["PENALTY_COEFF"]
                            ).mean(),
                            "actor_penalty_diff": action_diff.mean(),
                        }

                        return actor_loss, (updates, loss_infos)

                    # UPDATE CRITIC
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    (
                        critic_loss,
                        (batch_stats_update_critic, critic_loss_infos),
                    ), critic_grads = critic_grad_fn(
                        train_state["critic"].params, traj_batch, targets
                    )
                    train_state_critic = train_state["critic"].apply_gradients(
                        grads=critic_grads
                    )
                    train_state_critic = train_state_critic.replace(
                        grad_steps=train_state["critic"].grad_steps + 1,
                        batch_stats=batch_stats_update_critic["batch_stats"],
                    )

                    # UPDATE ACTOR
                    def update_actor(train_state_actor):
                        actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                        (
                            actor_loss,
                            (batch_stats_update_actor, actor_loss_infos),
                        ), actor_grads = actor_grad_fn(
                            train_state["actor"].params, traj_batch
                        )
                        train_state_actor = train_state["actor"].apply_gradients(
                            grads=actor_grads
                        )
                        train_state_actor = train_state_actor.replace(
                            grad_steps=train_state["actor"].grad_steps + 1,
                            batch_stats=batch_stats_update_actor["batch_stats"],
                        )
                        return (
                            train_state_actor,
                            actor_loss,
                            actor_grads,
                            actor_loss_infos,
                        )

                    # todo: add option to perform actor learning after several critic updates
                    train_state_actor, actor_loss, actor_grads, actor_loss_infos = (
                        update_actor(train_state["actor"])
                    )

                    # returns
                    train_state = {
                        "actor": train_state_actor,
                        "critic": train_state_critic,
                    }
                    total_loss = critic_loss + actor_loss
                    opt_infos = {
                        "critic_grads": critic_grads,
                        "actor_grads": actor_grads,
                        "critic_norm": optax.global_norm(critic_grads),
                        "actor_norm": optax.global_norm(actor_grads),
                    }
                    opt_infos.update(critic_loss_infos)
                    opt_infos.update(actor_loss_infos)
                    return train_state, (
                        total_loss,
                        (critic_loss, actor_loss, opt_infos),
                    )

                train_state, traj_batch, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, targets, rng)
                return update_state, loss_info

            update_state = (train_state, traj_batch, targets, rng)
            update_state, (total_loss, (value_loss, loss_actor, opt_infos)) = (
                jax.lax.scan(_update_epoch, update_state, None, config["NUM_EPOCHS"])
            )
            train_state = update_state[0]
            rng = update_state[-1]

            train_state["actor"] = train_state["actor"].replace(
                timesteps=(train_state["actor"].n_updates + 1)
                * config["NUM_ENVS"]
                * config["NUM_STEPS"],
                n_updates=train_state["actor"].n_updates + 1,
            )

            metrics = {
                "env_step": train_state["actor"].timesteps,
                "update_steps": train_state["actor"].n_updates,
                "grad_steps_actor": train_state["actor"].grad_steps,
                "grad_steps_critic": train_state["critic"].grad_steps,
                "noise": noise_scheduler(train_state["actor"].n_updates),
                "loss": total_loss.mean(),
                "value_loss": value_loss.mean(),
                "loss_actor": loss_actor.mean(),
                "lr": lr_scheduler(train_state["actor"].n_updates),
            }
            metrics.update({k: v.mean() for k, v in traj_batch.info.items()})
            jax.debug.print(
                "Step: {step}/{total_steps}",
                step=metrics["update_steps"],
                total_steps=config["NUM_UPDATES"],
            )

            opt_infos_metrics = {}
            for k, v in opt_infos.items():
                if isinstance(v, jax.Array):
                    opt_infos_metrics[f"opt_infos/{k}_mean"] = v.mean()
                    opt_infos_metrics[f"opt_infos/{k}_max"] = v.max()
                    opt_infos_metrics[f"opt_infos/{k}_min"] = v.min()
                elif isinstance(v, dict):
                    continue
            metrics.update(opt_infos_metrics)

            # test metrics
            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_state["actor"].n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(train_state, env_state, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_rng):
                    log_times.append(time.time())
                    if len(log_times) > 1:
                        dt = log_times[-1] - log_times[-2]
                        steps_per_update = (
                            config["NUM_ENVS"]
                            * config["NUM_STEPS"]
                            * config["NUM_SEEDS"]
                        )
                        metrics["sps"] = steps_per_update / dt
                        metrics["walltime"] = log_times[-1] - log_times[0]
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_rng)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics, original_rng)

            runner_state = (train_state, env_state, last_obs, rng, test_metrics)

            return_metrics = metrics if config.get("RETURN_METRICS", False) else None
            return runner_state, return_metrics

        def get_test_metrics(train_state, training_env_state, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                env_state, last_obs, rng, returns = carry
                rng, _rng = jax.random.split(rng)
                action = actor.apply(
                    {
                        "params": train_state["actor"].params,
                        "batch_stats": train_state["actor"].batch_stats,
                    },
                    last_obs["actor"],
                    train=False,
                )
                rng, _rng = jax.random.split(rng)
                _rng = jax.random.split(_rng, config["TEST_NUM_ENVS"])
                new_obs, new_env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                # increase returns only of running episodes, ignore ones already done
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

            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["TEST_NUM_ENVS"])
            init_obs, reset_env_state = env.reset(reset_rng, env_params)

            if config["NORMALIZE_OBS"]:
                env_state = training_env_state.replace(
                    env_state=reset_env_state.env_state,
                )
                init_obs["actor"] = (
                    init_obs["actor"] - env_state.actor_mean
                ) / jnp.sqrt(env_state.actor_var + 1e-8)
            else:
                env_state = reset_env_state

            returns = {
                "running_returns": jnp.zeros((config["TEST_NUM_ENVS"],)),
                "running_len": jnp.zeros((config["TEST_NUM_ENVS"],)),
                "running_done": jnp.zeros((config["TEST_NUM_ENVS"],), dtype=bool),
            }

            (new_env_state, new_obs, rng, returns), infos = jax.lax.scan(
                _env_step,
                (env_state, init_obs, _rng, returns),
                None,
                config["TEST_NUM_STEPS"],
            )
            done_infos = {
                "returned_episode_returns": returns["running_returns"].sum()
                / returns["running_done"].sum(),
                "returned_episode_lengths": returns["running_len"].sum()
                / returns["running_done"].sum(),
                "done_episodes": returns["running_done"].sum()
                / config["TEST_NUM_ENVS"],
            }
            return done_infos

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, env_state, _rng)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, test_metrics)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def single_run(config):

    config = {**config, **config["alg"]}

    alg_name = f'{config.get("ALG_NAME", "pqn")}_seed{config["SEED"]}'
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"Took {time.time()-t0} seconds to complete.")

    if config.get("SAVE_PATH", None) is not None:

        # define the output state
        train_state = outs["runner_state"][0]
        env_state = outs["runner_state"][1]  # Get the environment state
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
        save_name = f'{config["ALG_NAME"]}_{config["ENV_NAME"]}_seed{config["SEED"]}'
        save(to_save, config, save_dir, save_name, vmaps=config["NUM_SEEDS"])


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "ppo")
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
        "method": "grid",
        "metric": {
            "name": "test/returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR_START": {
                "min": 0.00001,
                "max": 0.01,
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    # wandb.agent(sweep_id, wrapped_make_train, count=1000)
    wandb.agent(
        sweep_id,
        function=wrapped_make_train,
        entity=default_config["ENTITY"],
        project=default_config["PROJECT"],
        count=1000,
    )


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
