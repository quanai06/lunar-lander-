from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import RecordVideo


@dataclass
class EnvInfo:
    env_id: str
    state_dim: int
    action_dim: int
    observation_shape: tuple[int, ...]
    is_discrete_action: bool


def make_env(
    env_id: str = "LunarLander-v3",
    render_mode: str | None = None,
) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    return env


def reset_env(env: gym.Env, seed: int | None = None) -> tuple[Any, dict]:
    obs, info = env.reset(seed=seed)
    return obs, info


def seed_env_spaces(env: gym.Env, seed: int) -> None:
    if hasattr(env, "action_space") and env.action_space is not None:
        env.action_space.seed(seed)

    if hasattr(env, "observation_space") and env.observation_space is not None:
        env.observation_space.seed(seed)


def get_env_info(env: gym.Env, env_id: str = "unknown") -> EnvInfo:
    obs_space = env.observation_space
    act_space = env.action_space

    if not hasattr(obs_space, "shape") or obs_space.shape is None:
        raise ValueError("Observation space must have a valid shape.")

    observation_shape = obs_space.shape
    state_dim = observation_shape[0]

    if hasattr(act_space, "n"):
        action_dim = act_space.n
        is_discrete_action = True
    elif hasattr(act_space, "shape") and act_space.shape is not None:
        action_dim = act_space.shape[0]
        is_discrete_action = False
    else:
        raise ValueError("Unsupported action space.")

    return EnvInfo(
        env_id=env_id,
        state_dim=state_dim,
        action_dim=action_dim,
        observation_shape=observation_shape,
        is_discrete_action=is_discrete_action,
    )


def wrap_env_for_video(
    env: gym.Env,
    video_folder: str | Path,
    episode_trigger=None,
) -> gym.Env:
    video_folder = Path(video_folder)
    video_folder.mkdir(parents=True, exist_ok=True)

    wrapped_env = RecordVideo(
        env,
        video_folder=str(video_folder),
        episode_trigger=episode_trigger,
    )
    return wrapped_env


def close_env(env: gym.Env) -> None:
    env.close()