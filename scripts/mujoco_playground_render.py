# use this script to render videos of trained policies in playground environments

import jax
from jax import numpy as jnp

import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


from purejaxql.utils.brax_wrappers import (
    get_original_state,
    PlaygroundVecGymnaxWrapper,
    LogVecWrapper,
    ClipAction,
    NormalizeVecReward,
    NormalizeVecObservation,
)
from purejaxql.pqn_mujoco_playground import Actor
from purejaxql.utils.save_load import load_params
from omegaconf import OmegaConf


import mujoco
import mediapy as media
import argparse
import optax
from typing import Any, NamedTuple


# custom command for locomotion environments
x_vel = 1.0  # @param {type: "number"}
y_vel = 0.0  # @param {type: "number"}
yaw_vel = 0.0  # @param {type: "number"}
CUSTOM_LOCOMOTION_COMMAND = jnp.array([x_vel, y_vel, yaw_vel])


class InferenceModelState(NamedTuple):
    params: Any
    batch_stats: Any
    normalization_stats: Any = None


def preprocess_obs(obs, normalization_stats):
    if normalization_stats is not None:
        obs["actor"] = (obs["actor"] - normalization_stats["actor_mean"]) / jnp.sqrt(
            normalization_stats["actor_var"] + 1e-8
        )
        obs["critic"] = (obs["critic"] - normalization_stats["critic_mean"]) / jnp.sqrt(
            normalization_stats["critic_var"] + 1e-8
        )
    return obs


def get_test_metrics(
    actor, env, train_state, seed=0, num_envs=1, num_steps=1000, custom_command=True
):

    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, num_envs)
    init_obs, env_state = env.reset(reset_rng, None)

    returns = {
        "running_returns": jnp.zeros((num_envs,)),
        "running_lengths": jnp.zeros((num_envs,)),
        "running_done": jnp.zeros((num_envs,), dtype=bool),
    }

    state = get_original_state(env_state)
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )  # pytype: disable=attribute-error
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})
    empty_traj = empty_traj.replace(data=empty_data)

    def _env_step(carry, _):
        env_state, last_obs, rng, returns = carry

        last_obs = preprocess_obs(last_obs, train_state.normalization_stats)
        rng, _rng = jax.random.split(rng)
        action = actor.apply(
            {
                "params": train_state.params,
                "batch_stats": train_state.batch_stats,
            },
            last_obs["actor"],
            train=False,
        )
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, num_envs)
        new_obs, new_env_state, reward, done, info = env.step(
            _rng, env_state, action, None
        )

        # increase returns only of running episodes, ignore ones already done
        returns["running_returns"] = jnp.where(
            ~returns["running_done"],
            returns["running_returns"] + info["original_reward"],
            returns["running_returns"],
        )
        returns["running_lengths"] = jnp.where(
            ~returns["running_done"],
            returns["running_lengths"] + 1,
            returns["running_lengths"],
        )
        returns["running_done"] = (returns["running_done"] + done).astype(bool)

        state = get_original_state(env_state)
        traj_data = empty_traj.tree_replace(
            {
                "data.qpos": state.data.qpos,
                "data.qvel": state.data.qvel,
                "data.time": state.data.time,
                "data.ctrl": state.data.ctrl,
                "data.mocap_pos": state.data.mocap_pos,
                "data.mocap_quat": state.data.mocap_quat,
                "data.xfrc_applied": state.data.xfrc_applied,
            }
        )

        return (new_env_state, new_obs, rng, returns), (traj_data, info)

    (new_env_state, new_obs, rng, returns), (traj_data, infos) = jax.lax.scan(
        _env_step,
        (env_state, init_obs, _rng, returns),
        None,
        num_steps,
    )
    done_infos = {
        "returned_episode_returns": returns["running_returns"].sum()
        / returns["running_done"].sum(),
        "done_episodes": returns["running_done"].sum() / num_envs,
        "returned_episode_lengths": returns["running_lengths"].sum()
        / returns["running_done"].sum(),
    }
    returns = returns["running_returns"]
    return traj_data, done_infos, returns


def get_data(
    train_state,
    config,
    env,
    num_episodes=2,
    episode_length=1000,
    seed=0,
):

    # initialize actor with environment action space
    action_space = env.action_space(None)
    actor = Actor(
        action_space.shape[0],
        action_scale=jnp.array((action_space.high - action_space.low) / 2.0),
        action_bias=jnp.array((action_space.high + action_space.low) / 2.0),
        hidden_sizes=config["ACTOR_HIDDEN_SIZES"],
        activation=config.get("ACTIVATION", "relu"),
        norm_type=config["NORM_TYPE"],
        init_scale=config.get("ACTOR_INIT_SCALE", 0.01),
    )
    print(f"action scale: {actor.action_scale}, action bias: {actor.action_bias}")

    traj_data, done_infos, returns = get_test_metrics(
        actor=actor,
        env=env,
        train_state=train_state,
        seed=seed,
        num_envs=num_episodes,
        num_steps=episode_length,
    )

    print(done_infos)

    return traj_data, returns


def render(
    model_path,
    config_path,
    output_path="renders",
    num_videos=2,
    env_name=None,
    episode_length=None,
    camera="track",
):

    print("Model Path:", model_path)
    print("Config Path:", config_path)

    # config and env
    config = OmegaConf.load(config_path)

    if env_name is None:
        env_name = config["ENV_NAME"]

    print("Config:", config)
    print("Env Name:", env_name)

    # define custom command for playground envs, for example to move the agent in one direction
    use_custom_command = False
    if use_custom_command:
        custom_command = CUSTOM_LOCOMOTION_COMMAND
    else:
        custom_command = None

    env, env_params = (
        PlaygroundVecGymnaxWrapper(config["ENV_NAME"], custom_command=custom_command),
        None,
    )
    episode_length = env.episode_length if episode_length is None else episode_length
    print("episode_length:", episode_length)
    action_space = env.action_space(None)
    env = LogVecWrapper(env)
    env = ClipAction(
        env,
        low=action_space.low,
        high=action_space.high,
    )

    # actor
    if model_path is None:
        # init as random
        action_space = env.action_space(None)
        actor = Actor(
            action_space.shape[0],
            action_scale=jnp.array((action_space.high - action_space.low) / 2.0),
            action_bias=jnp.array((action_space.high + action_space.low) / 2.0),
            hidden_sizes=config["ACTOR_HIDDEN_SIZES"],
            activation=config.get("ACTIVATION", "relu"),
            norm_type=config["NORM_TYPE"],
            init_scale=config.get("ACTOR_INIT_SCALE", 1.0),
        )
        rng = jax.random.PRNGKey(0)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params)["actor"].shape)
        actor_variables = actor.init(_rng, init_x)
        model_state = InferenceModelState(
            params=actor_variables["params"],
            batch_stats=actor_variables["batch_stats"],
            normalization_stats=None,
        )
    else:
        print("loading pretrained params")
        variables = load_params(model_path)
        normalization_stats = variables.get("normalization_stats", None)
        model_state = InferenceModelState(
            params=variables["actor"]["params"],
            batch_stats=variables["actor"]["batch_stats"],
            normalization_stats=normalization_stats,
        )

    print(jax.tree.map(lambda x: x.shape, model_state))

    traj_stacked, returns = get_data(
        model_state,
        config,
        env,
        num_episodes=num_videos,
        episode_length=episode_length,
        seed=0,
    )

    trajectories = [None] * num_videos
    trajectories = []
    for i in range(num_videos):
        t = jax.tree.map(lambda x, i=i: x[:, i], traj_stacked)
        traj = [
            jax.tree.map(lambda arr, t_idx=t_idx: arr[t_idx], t)
            for t_idx in range(episode_length)
        ]
        trajectories.append(traj)

    # Render and save the rollout.
    os.makedirs(output_path, exist_ok=True)
    render_every = 2
    fps = 1.0 / env.dt / render_every
    print(f"FPS for rendering: {fps}")
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    for i, rollout in enumerate(trajectories):
        traj = rollout[::render_every]
        try:
            frames = env.render(
                traj,
                camera=camera,
                height=480,
                width=640,
                scene_option=scene_option,
            )
        except Exception as e:
            print(f"Rendering failed: {e}, trying with default camera...")
            frames = env.render(
                traj,
                camera=None,
                height=480,
                width=640,
                scene_option=scene_option,
            )
        out_path = f"{output_path}/{env_name}_rollout{i}_returns{int(returns[i])}.mp4"
        media.write_video(
            out_path,
            frames,
            fps=fps,
        )
        print(f"Rollout video saved as '{out_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Render rollout videos using a trained model"
    )
    parser.add_argument(
        "--model_path",
        default="models_playground/CartpoleBalance/pqn_CartpoleBalance_seed0_vmap0.safetensors",
        type=str,
        help="Path to the trained model parameters file",
    )
    parser.add_argument(
        "--config_path",
        default="models_playground/CartpoleBalance/pqn_CartpoleBalance_seed0_config.yaml",
        type=str,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--output_path",
        default="renders",
        type=str,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--num_videos", "-n", type=int, default=2, help="Number of videos to render"
    )
    parser.add_argument(
        "--env_name",
        "-e",
        type=str,
        default=None,
        help="Environment name (overrides config ENV_NAME)",
    )
    parser.add_argument(
        "--episode_length",
        "-l",
        type=int,
        default=None,
        help="Episode length (overrides config TEST_NUM_STEPS)",
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=str,
        default="track",
        help="Camera option (leave blank for fixed, otherwise 'track')",
    )
    parser.add_argument(
        "--custom_command",
        "-cc",
        type=bool,
        default=False,
        help="Use custom command for locomotion envs (default: False)",
    )
    args = parser.parse_args()
    render(
        args.model_path,
        args.config_path,
        output_path=args.output_path,
        num_videos=args.num_videos,
        env_name=args.env_name,
        episode_length=args.episode_length,
        camera=args.camera,
    )


if __name__ == "__main__":
    main()
