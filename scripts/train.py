from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.dqn_agent import DQNAgent
from src.envs import (
    close_env,
    get_env_info,
    make_env,
    seed_env_spaces,
    wrap_env_for_video,
)
from src.trainers.trainer import Trainer
from src.utils.seed import set_seed


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return requested_device


def record_agent_video(
    agent: DQNAgent,
    env_id: str,
    video_dir: str | Path,
    video_prefix: str,
    seed: int | None = None,
    max_steps: int = 1000,
) -> None:
    """
    Record one evaluation episode to video.

    Args:
        agent: Trained or untrained agent.
        env_id: Gymnasium environment id.
        video_dir: Directory to save videos.
        video_prefix: Prefix to separate before/after training videos.
        seed: Optional seed.
        max_steps: Max steps for the recorded episode.
    """
    video_dir = Path(video_dir) / video_prefix
    raw_env = make_env(env_id=env_id, render_mode="rgb_array")
    video_env = wrap_env_for_video(
        raw_env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,
    )

    try:
        state, _ = video_env.reset(seed=seed)

        for _ in range(max_steps):
            action = agent.act(state, training=False)
            next_state, _, terminated, truncated, _ = video_env.step(action)
            state = next_state

            if terminated or truncated:
                break
    finally:
        close_env(video_env)


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "dqn.yaml")

    env_cfg = config["env"]
    agent_cfg = config["agent"]
    trainer_cfg = config["trainer"]
    paths_cfg = config["paths"]

    seed = env_cfg["seed"]
    set_seed(seed)

    device = resolve_device(agent_cfg["device"])

    # Main training env
    env = make_env(
        env_id=env_cfg["env_id"],
        render_mode=env_cfg["render_mode"],
    )
    seed_env_spaces(env, seed)

    env_info = get_env_info(env, env_id=env_cfg["env_id"])

    agent = DQNAgent(
        state_dim=env_info.state_dim,
        action_dim=env_info.action_dim,
        buffer_capacity=agent_cfg["buffer_capacity"],
        hidden_dims=tuple(agent_cfg["hidden_dims"]),
        learning_rate=agent_cfg["learning_rate"],
        gamma=agent_cfg["gamma"],
        batch_size=agent_cfg["batch_size"],
        epsilon_start=agent_cfg["epsilon_start"],
        epsilon_end=agent_cfg["epsilon_end"],
        epsilon_decay=agent_cfg["epsilon_decay"],
        target_update_freq=agent_cfg["target_update_freq"],
        device=device,
    )

    trainer = Trainer(
        env=env,
        agent=agent,
        num_episodes=trainer_cfg["num_episodes"],
        max_steps_per_episode=trainer_cfg["max_steps_per_episode"],
        warmup_steps=trainer_cfg["warmup_steps"],
        checkpoint_dir=paths_cfg["checkpoint_dir"],
        save_every=trainer_cfg["save_every"],
        checkpoint_name="dqn_agent.pt",
    )

    try:
        # Record before training
        print("Recording video before training...")
        record_agent_video(
            agent=agent,
            env_id=env_cfg["env_id"],
            video_dir=paths_cfg["video_dir"],
            video_prefix="before_training",
            seed=seed,
            max_steps=trainer_cfg["max_steps_per_episode"],
        )

        # Train
        history = trainer.train(seed=seed, verbose=True)
        print(f"Training finished. Total episodes: {len(history)}")

        # Evaluate metrics
        eval_metrics = trainer.evaluate(num_episodes=5, seed=seed)
        print("Evaluation results:")
        for key, value in eval_metrics.items():
            print(f"{key}: {value:.4f}")

        # Record after training
        print("Recording video after training...")
        record_agent_video(
            agent=agent,
            env_id=env_cfg["env_id"],
            video_dir=paths_cfg["video_dir"],
            video_prefix="after_training",
            seed=seed,
            max_steps=trainer_cfg["max_steps_per_episode"],
        )

    finally:
        close_env(env)


if __name__ == "__main__":
    main()