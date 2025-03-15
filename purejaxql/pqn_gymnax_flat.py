"""*Exactly* the same algorithm as `pqn_gymnax.py`, but with typing and reduced nesting."""

import copy
import operator
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generic, Literal

import chex
import flax
import flax.struct
import gymnax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.typing import FrozenVariableDict
from gymnax.environments.environment import Environment, TEnvParams, TEnvState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from omegaconf import OmegaConf
from safetensors.flax import load_file, save_file
from typing_extensions import NotRequired, TypedDict
from xtils.jitpp import Static, jit

import wandb

# Reuse the network and train state classes from the original script.
from purejaxql.pqn_gymnax import CustomTrainState, QNetwork


class Transition(flax.struct.PyTreeNode):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    next_obs: jax.Array
    q_val: jax.Array


class Config(TypedDict):
    SEED: int
    NUM_SEEDS: int
    TOTAL_TIMESTEPS: int
    NUM_STEPS: int
    NUM_ENVS: int
    TOTAL_TIMESTEPS_DECAY: int
    NUM_ENVS: int
    NUM_MINIBATCHES: int
    NUM_EPOCHS: int
    ENV_NAME: str
    EPS_START: float
    EPS_FINISH: float
    EPS_DECAY: int
    EPS_TEST: float
    LR: float
    NORM_TYPE: Literal["layer_norm", "batch_norm"] | None
    NORM_INPUT: NotRequired[bool]
    NORM_LAYERS: NotRequired[int]
    TEST_NUM_ENVS: int
    TEST_INTERVAL: int
    MAX_GRAD_NORM: float
    LAMBDA: float
    GAMMA: float
    REW_SCALE: NotRequired[float]
    HIDDEN_SIZE: NotRequired[int]
    NUM_LAYERS: NotRequired[int]
    ENTITY: str
    PROJECT: str
    WANDB_MODE: Literal["disabled", "online", "offline"]
    ALG_NAME: str
    SAVE_PATH: NotRequired[str]
    WANDB_LOG_ALL_SEEDS: NotRequired[bool]
    HYP_TUNE: bool


class Results(TypedDict, Generic[TEnvState]):
    runner_state: tuple[
        CustomTrainState, tuple[jax.Array, TEnvState], Any, chex.PRNGKey
    ]
    metrics: dict[str, jax.Array]


def _get_num_updates(config: Config) -> int:
    return config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]


def _get_num_updates_decay(config: Config) -> int:
    return config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]


def _get_test_num_steps(config: Config, env_params: gymnax.EnvParams) -> int:
    return config.get("TEST_NUM_STEPS", env_params.max_steps_in_episode)


def make_train(config: Config):
    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    return partial(
        train,
        config=FrozenDict(config),
        env=env,
        env_params=env_params,
        num_updates=_get_num_updates(config),
        num_updates_decay=_get_num_updates_decay(config),
        test_num_steps=_get_test_num_steps(config, env_params=env_params),
    )


@jit
def train(
    rng: chex.PRNGKey,
    config: Static[Config],
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
    num_updates: Static[int],
    num_updates_decay: Static[int],
    test_num_steps: Static[int],
) -> Results[TEnvState]:
    # todo: Why at index 0? Is this assuming that we're always under `vmap` context?
    # Seems like this is just used to get an actual 'int'-like value that we can use
    # as a key in the metrics dict when using the WANDB_LOG_ALL_SEEDS option.
    original_rng = rng[0]
    num_envs: int = config["NUM_ENVS"]
    test_num_envs: int = config["TEST_NUM_ENVS"]

    eps_scheduler = optax.linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        config["EPS_DECAY"] * num_updates_decay,
    )

    lr_scheduler = optax.linear_schedule(
        init_value=config["LR"],
        end_value=1e-20,
        transition_steps=num_updates_decay
        * config["NUM_MINIBATCHES"]
        * config["NUM_EPOCHS"],
    )
    lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

    # INIT NETWORK AND OPTIMIZER
    network = QNetwork(
        action_dim=env.action_space(env_params).n,
        hidden_size=config.get("HIDDEN_SIZE", 128),
        num_layers=config.get("NUM_LAYERS", 2),
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
    )
    rng, _rng = jax.random.split(rng)
    train_state = create_agent(
        rng, env=env, env_params=env_params, config=config, network=network, lr=lr
    )

    # TRAINING LOOP
    rng, _rng = jax.random.split(rng)
    test_metrics = _get_test_metrics(
        train_state,
        _rng,
        env=env,
        env_params=env_params,
        config=config,
        network=network,
        test_num_envs=test_num_envs,
        test_num_steps=test_num_steps,
    )

    rng, _rng = jax.random.split(rng)
    expl_state = _vmap_reset(_rng, n_envs=num_envs, env=env, env_params=env_params)

    # train
    rng, _rng = jax.random.split(rng)

    runner_state = (train_state, expl_state, test_metrics, _rng)
    runner_state, metrics = jax.lax.scan(
        lambda runner_state, _input: _update_step(
            runner_state,
            _input,
            network=network,
            num_envs=num_envs,
            eps_scheduler=eps_scheduler,
            config=config,
            env=env,
            env_params=env_params,
            original_rng=original_rng,
            test_num_steps=test_num_steps,
            test_num_envs=test_num_envs,
        ),
        init=runner_state,
        xs=jnp.arange(num_updates),
        length=num_updates,
    )

    return {"runner_state": runner_state, "metrics": metrics}


@jit
def _get_test_metrics(
    train_state: CustomTrainState,
    rng: chex.PRNGKey,
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
    config: Static[Config],
    network: Static[QNetwork],
    test_num_envs: Static[int],
    test_num_steps: Static[int],
):
    test_during_training: bool = config.get("TEST_DURING_TRAINING", False)
    # test_num_envs: int = config["TEST_NUM_ENVS"]
    test_epsilon: float = config["EPS_TEST"]
    if not test_during_training:
        return None

    def _env_step(
        carry: tuple[TEnvState, jax.Array, chex.PRNGKey], _: Any
    ) -> tuple[tuple[TEnvState, jax.Array, chex.PRNGKey], dict]:
        env_state, last_obs, rng = carry
        rng, _rng = jax.random.split(rng)
        q_vals = network.apply(
            {
                "params": train_state.params,
                "batch_stats": train_state.batch_stats,
            },
            last_obs,
            train=False,
        )
        eps = jnp.full(test_num_envs, test_epsilon)
        assert isinstance(q_vals, jax.Array)
        action = jax.vmap(eps_greedy_exploration)(
            jax.random.split(_rng, test_num_envs), q_vals, eps
        )
        new_obs, new_env_state, reward, done, info = _vmap_step(
            _rng,
            env_state=env_state,
            action=action,
            env=env,
            env_params=env_params,
            n_envs=test_num_envs,
        )
        assert isinstance(new_obs, jax.Array)
        return (new_env_state, new_obs, rng), info  # type: ignore

    rng, _rng = jax.random.split(rng)
    _rng: chex.PRNGKey
    init_obs, env_state = _vmap_reset(
        _rng, n_envs=test_num_envs, env=env, env_params=env_params
    )
    # init_obs, env_state = vmap_reset(test_num_envs)(_rng)

    _, infos = jax.lax.scan(
        _env_step, (env_state, init_obs, _rng), xs=None, length=test_num_steps
    )
    # return mean of done infos
    returned_episode: jax.Array = infos["returned_episode"]

    def _get_mean_of_done_episode_infos(x: jax.Array) -> jax.Array:
        return jnp.nanmean(jnp.where(returned_episode, x, jnp.nan))

    done_infos = jax.tree.map(_get_mean_of_done_episode_infos, infos)
    return done_infos


@jit
def _update_step(
    runner_state: tuple[
        CustomTrainState, tuple[jax.Array, TEnvState], Any, chex.PRNGKey
    ],
    _update_step_index: jax.Array,
    network: Static[QNetwork],
    num_envs: Static[int],
    eps_scheduler: Static[optax.Schedule],
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
    config: Static[Config],
    original_rng: chex.PRNGKey,
    test_num_steps: Static[int],
    test_num_envs: Static[int],
):
    train_state, expl_state, test_metrics, rng = runner_state
    _rng: chex.PRNGKey

    # SAMPLE PHASE

    # step the env
    rng, _rng = jax.random.split(rng)
    obs, env_state = expl_state
    reward_scaling_coef: float = config.get("REW_SCALE", 1)

    step_env = partial(
        _train_env_step,
        network=network,
        train_state=train_state,
        num_envs=num_envs,
        eps_scheduler=eps_scheduler,
        env=env,
        env_params=env_params,
        reward_scaling_coef=reward_scaling_coef,
    )

    (obs, env_state, rng), (transitions, infos) = jax.lax.scan(
        step_env,
        init=(obs, env_state, _rng),
        xs=None,
        length=config["NUM_STEPS"],
    )
    expl_state = (obs, env_state)
    # update timesteps count
    train_state = train_state.replace(
        timesteps=train_state.timesteps + config["NUM_STEPS"] * num_envs
    )

    last_q = network.apply(
        {
            "params": train_state.params,
            "batch_stats": train_state.batch_stats,
        },
        transitions.next_obs[-1],
        train=False,
    )
    assert isinstance(last_q, jax.Array)
    last_q = jnp.max(last_q, axis=-1)

    last_q = last_q * (1 - transitions.done[-1])
    lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
    _, targets = jax.lax.scan(
        partial(_get_target, gamma=config["GAMMA"], lambda_=config["LAMBDA"]),
        (lambda_returns, last_q),
        jax.tree.map(lambda x: x[:-1], transitions),
        reverse=True,
    )
    lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

    # NETWORKS UPDATE
    _learn_epoch = partial(
        learn_epoch,
        network=network,
        num_minibatches=config["NUM_MINIBATCHES"],
        transitions=transitions,
        lambda_targets=lambda_targets,
    )
    rng, _rng = jax.random.split(rng)
    (train_state, rng), (loss, qvals) = jax.lax.scan(
        _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
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
        num_updates = _get_num_updates(config)
        rng, _rng = jax.random.split(rng)
        test_metrics = jax.lax.cond(
            train_state.n_updates % int(num_updates * config["TEST_INTERVAL"]) == 0,
            lambda _: _get_test_metrics(
                train_state,
                _rng,
                env=env,
                env_params=env_params,
                config=config,
                network=network,
                test_num_steps=test_num_steps,
                test_num_envs=test_num_envs,
            ),
            lambda _: test_metrics,
            operand=None,
        )
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    # report on wandb if required
    if config["WANDB_MODE"] != "disabled":

        def callback(metrics: dict, original_rng: chex.PRNGKey):
            if config.get("WANDB_LOG_ALL_SEEDS", False):
                metrics.update(
                    {f"rng{int(original_rng)}/{k}": v for k, v in metrics.items()}
                )
            wandb.log(metrics, step=metrics["update_steps"])

        jax.debug.callback(callback, metrics, original_rng)

    runner_state = (train_state, expl_state, test_metrics, rng)

    return runner_state, metrics


@jit
def _get_target(
    lambda_returns_and_next_q: tuple[jax.Array, jax.Array],
    transition: Transition,
    gamma: float,
    lambda_: float,
):
    lambda_returns, next_q = lambda_returns_and_next_q
    target_bootstrap = transition.reward + gamma * (1 - transition.done) * next_q
    delta = lambda_returns - next_q
    lambda_returns = target_bootstrap + gamma * lambda_ * delta
    # note: what about?
    # lambda_returns = jnp.where(transition.done, transition.reward, lambda_returns)
    lambda_returns = (
        1 - transition.done
    ) * lambda_returns + transition.done * transition.reward
    next_q = jnp.max(transition.q_val, axis=-1)
    return (lambda_returns, next_q), lambda_returns


@jit
def learn_epoch(
    carry: tuple[CustomTrainState, chex.PRNGKey],
    _,
    network: Static[QNetwork],
    num_minibatches: Static[int],
    transitions: Transition,
    lambda_targets: jax.Array,
):
    train_state, rng = carry

    rng, _rng = jax.random.split(rng)
    # num_actors*num_envs (batch_size), ...
    preprocess_fn = partial(
        preprocess_transition, rng=_rng, num_minibatches=num_minibatches
    )
    minibatches: Transition = jax.tree.map(preprocess_fn, transitions)
    targets: jax.Array = jax.tree.map(preprocess_fn, lambda_targets)

    rng, _rng = jax.random.split(rng)
    (train_state, rng), (loss, qvals) = jax.lax.scan(
        partial(_learn_phase, network=network),
        (train_state, rng),
        (minibatches, targets),
    )

    return (train_state, rng), (loss, qvals)


@jit
def preprocess_transition(
    x: jax.Array, rng: chex.PRNGKey, num_minibatches: Static[int]
):
    x = x.reshape(-1, *x.shape[2:])  # num_steps*num_envs (batch_size), ...
    x = jax.random.permutation(rng, x)  # shuffle the transitions
    x = x.reshape(
        # config["NUM_MINIBATCHES"], -1, *x.shape[1:]
        num_minibatches,
        -1,
        *x.shape[1:],
    )  # num_mini_updates, batch_size/num_mini_updates, ...
    return x


@jit
def _learn_phase(
    carry: tuple[CustomTrainState, chex.PRNGKey],
    minibatch_and_target: tuple[Transition, jax.Array],
    network: Static[QNetwork],
):
    train_state, rng = carry
    minibatch, target = minibatch_and_target

    def _loss_fn(params: FrozenVariableDict):
        q_vals, updates = network.apply(
            {"params": params, "batch_stats": train_state.batch_stats},
            minibatch.obs,
            train=True,
            mutable=["batch_stats"],
        )  # (batch_size*2, num_actions)

        chosen_action_qvals = jnp.take_along_axis(
            q_vals,
            jnp.expand_dims(minibatch.action, axis=-1),
            axis=-1,
        ).squeeze(axis=-1)

        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

        return loss, (updates, chosen_action_qvals)

    (loss, (updates, qvals)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        train_state.params
    )
    train_state = train_state.apply_gradients(grads=grads)
    train_state = train_state.replace(
        grad_steps=train_state.grad_steps + 1,
        batch_stats=updates["batch_stats"],
    )
    return (train_state, rng), (loss, qvals)


def _train_env_step(  # aka '_step_env' in main.
    carry: tuple[jax.Array, gymnax.EnvState, chex.PRNGKey],
    _,
    network: Static[QNetwork],
    train_state: CustomTrainState,
    num_envs: Static[int],
    eps_scheduler: Static[Callable[[int], float]],
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
    reward_scaling_coef: float,
):
    last_obs, env_state, rng = carry
    rng, rng_a, rng_s = jax.random.split(rng, 3)
    q_vals = network.apply(
        {
            "params": train_state.params,
            "batch_stats": train_state.batch_stats,
        },
        last_obs,
        train=False,
    )
    assert isinstance(q_vals, jax.Array)

    # different eps for each env
    _rngs = jax.random.split(rng_a, num_envs)
    eps = jnp.full(num_envs, eps_scheduler(train_state.n_updates))
    new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

    new_obs, new_env_state, reward, new_done, info = _vmap_step(
        rng_s,
        env_state=env_state,
        action=new_action,
        n_envs=num_envs,
        env=env,
        env_params=env_params,
    )

    transition = Transition(
        obs=last_obs,
        action=new_action,
        reward=reward_scaling_coef * reward,
        # reward=config.get("REW_SCALE", 1) * reward,
        done=new_done,
        next_obs=new_obs,
        q_val=q_vals,
    )
    assert isinstance(new_obs, jax.Array)
    rng: chex.PRNGKey
    return (new_obs, new_env_state, rng), (transition, info)


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


@jit
def create_agent(
    rng: chex.PRNGKey,
    env: Static[Environment],
    env_params: gymnax.EnvParams,
    config: Static[Config],
    network: Static[QNetwork],
    lr: Static[optax.Schedule | float],
):
    init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
    network_variables = network.init(rng, init_x, train=False)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.radam(learning_rate=lr),
    )

    train_state = CustomTrainState.create(
        apply_fn=network.apply,
        params=network_variables["params"],
        batch_stats=network_variables["batch_stats"],
        tx=tx,
    )
    return train_state


def eps_greedy_exploration(rng: chex.PRNGKey, q_vals: jax.Array, eps: jax.Array):
    # a key for sampling random actions and one for picking
    rng_a, rng_e = jax.random.split(rng)
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    chosen_actions = jnp.where(
        # pick the actions that should be random
        jax.random.uniform(rng_e, greedy_actions.shape) < eps,
        # sample random actions,
        jax.random.randint(
            rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
        ),
        greedy_actions,
    )
    return chosen_actions


# Including this here to make this compatible with jaxmarl 0.0.4
# (seems like those functions were removed or moved to a different place?)


def save_params(params: dict, filename: str | os.PathLike) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)  # type: ignore


def load_params(filename: str | os.PathLike) -> dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


def single_run(_config: dict):
    config: Config = {**_config, **_config["alg"]}  # type: ignore

    alg_name: str = config.get("ALG_NAME", "pqn")
    env_name: str = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config.get("NAME", f'{config["ALG_NAME"]}_{config["ENV_NAME"]}'),
        config=dict(config),
        mode=config["WANDB_MODE"],
        save_code=True,
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.perf_counter()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))  # type: ignore
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"Took {time.perf_counter()-t0:.2f} seconds to complete.")

    if (save_path := config.get("SAVE_PATH")) is not None:
        # todo: this import is failing:
        model_state = outs["runner_state"][0]
        save_dir = Path(save_path) / env_name
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            dict(config),
            save_dir / f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml',
        )
        for i, rng in enumerate(rngs):
            params = jax.tree.map(operator.itemgetter(i), model_state.params)
            save_path = (
                save_dir
                / f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors'
            )
            save_params(params, save_path)
    return outs


def tune(_default_config):
    """Hyperparameter sweep with wandb."""

    default_config: Config = {**_default_config, **_default_config["alg"]}  # type: ignore
    alg_name = default_config.get("ALG_NAME", "pqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        # TODO: Weird, why is this doing one training run per item in wandb.config?
        # Is there only one key being tuned?
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        _outs = jax.block_until_ready(train_vjit(rngs))

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
    print("Config:\n", OmegaConf.to_yaml(config))
    assert isinstance(config, dict)
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
