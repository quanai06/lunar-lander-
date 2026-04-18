from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.grpo_agent import GRPOAgent
from src.agents.ppo_agent import PPOAgent
from src.envs import close_env, get_env_info, make_env, seed_env_spaces
from src.trainers.on_policy_trainer import OnPolicyTrainer
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


def make_on_policy_agent(
    algorithm: str,
    state_dim: int,
    action_dim: int,
    shared_cfg: dict,
    override_cfg: dict,
) -> PPOAgent | GRPOAgent:
    hidden_dims = tuple(override_cfg.get("hidden_dims", shared_cfg["hidden_dims"]))
    learning_rate = float(override_cfg.get("learning_rate", shared_cfg["learning_rate"]))
    gamma = float(override_cfg.get("gamma", shared_cfg["gamma"]))
    gae_lambda = float(override_cfg.get("gae_lambda", shared_cfg["gae_lambda"]))
    clip_eps = float(override_cfg.get("clip_eps", shared_cfg["clip_eps"]))
    entropy_coef = float(override_cfg.get("entropy_coef", shared_cfg["entropy_coef"]))
    value_coef = float(override_cfg.get("value_coef", shared_cfg["value_coef"]))
    update_epochs = int(override_cfg.get("update_epochs", shared_cfg["update_epochs"]))
    mini_batch_size = int(override_cfg.get("mini_batch_size", shared_cfg["mini_batch_size"]))
    max_grad_norm = float(override_cfg.get("max_grad_norm", shared_cfg["max_grad_norm"]))
    group_size = int(override_cfg.get("group_size", shared_cfg["group_size"]))
    device = str(override_cfg.get("device", shared_cfg["device"]))

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dims": hidden_dims,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_eps": clip_eps,
        "entropy_coef": entropy_coef,
        "value_coef": value_coef,
        "update_epochs": update_epochs,
        "mini_batch_size": mini_batch_size,
        "max_grad_norm": max_grad_norm,
        "group_size": group_size,
        "device": resolve_device(device),
    }

    if algorithm == "ppo":
        return PPOAgent(**kwargs)
    if algorithm == "grpo":
        return GRPOAgent(**kwargs)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "ppo_grpo_benchmark.yaml")

    env_cfg = config["env"]
    shared_agent_cfg = config["shared_agent"]
    trainer_cfg = config["trainer"]
    benchmark_cfg = config["benchmark"]
    paths_cfg = config["paths"]

    seed = int(env_cfg["seed"])
    set_seed(seed)

    algorithms = benchmark_cfg["algorithms"]
    eval_episodes = int(benchmark_cfg["eval_episodes"])
    train_reward_window = int(benchmark_cfg["train_reward_window"])
    verbose = bool(benchmark_cfg.get("verbose", True))

    algorithm_overrides = config.get("algorithm_overrides", {})
    rows: list[dict[str, float | int | str]] = []

    checkpoint_root = Path(paths_cfg["checkpoint_root"])
    csv_path = Path(paths_cfg["benchmark_csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    for algorithm in algorithms:
        print(f"\n=== Running {algorithm.upper()} ===")
        set_seed(seed)
        override_cfg = algorithm_overrides.get(algorithm, {})
        env = make_env(env_id=env_cfg["env_id"], render_mode=env_cfg["render_mode"])

        try:
            seed_env_spaces(env, seed)
            env_info = get_env_info(env, env_id=env_cfg["env_id"])

            agent = make_on_policy_agent(
                algorithm=algorithm,
                state_dim=env_info.state_dim,
                action_dim=env_info.action_dim,
                shared_cfg=shared_agent_cfg,
                override_cfg=override_cfg,
            )

            checkpoint_dir = checkpoint_root / algorithm
            trainer = OnPolicyTrainer(
                env=env,
                agent=agent,
                num_episodes=int(trainer_cfg["num_episodes"]),
                max_steps_per_episode=int(trainer_cfg["max_steps_per_episode"]),
                checkpoint_dir=checkpoint_dir,
                save_every=int(trainer_cfg["save_every"]),
                checkpoint_name=f"{algorithm}_agent.pt",
            )

            history = trainer.train(seed=seed, verbose=verbose)
            eval_metrics = trainer.evaluate(num_episodes=eval_episodes, seed=seed)

            train_rewards = np.array([stat.episode_reward for stat in history], dtype=np.float32)
            window_rewards = train_rewards[-train_reward_window:]

            row = {
                "algorithm": algorithm,
                "episodes": int(trainer_cfg["num_episodes"]),
                "train_mean_reward_last_window": float(np.mean(window_rewards)),
                "train_std_reward_last_window": float(np.std(window_rewards)),
                "eval_mean_reward": eval_metrics["mean_reward"],
                "eval_std_reward": eval_metrics["std_reward"],
                "eval_mean_length": eval_metrics["mean_length"],
            }
            rows.append(row)

            print(
                f"{algorithm.upper()} eval | "
                f"mean_reward={row['eval_mean_reward']:.2f}, "
                f"std_reward={row['eval_std_reward']:.2f}, "
                f"mean_length={row['eval_mean_length']:.2f}"
            )
        finally:
            close_env(env)

    benchmark_df = pd.DataFrame(rows)
    benchmark_df.to_csv(csv_path, index=False)

    print("\nBenchmark completed.")
    print(f"Saved benchmark table to: {csv_path}")
    print(benchmark_df.to_string(index=False))


if __name__ == "__main__":
    main()
