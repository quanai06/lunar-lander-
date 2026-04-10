from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.dqn_agent import DQNAgent


@dataclass
class TrainStats:
    episode: int
    episode_reward: float
    episode_length: int
    average_loss: float | None
    epsilon: float


class Trainer:
    def __init__(
        self,
        env: Any,
        agent: DQNAgent,
        num_episodes: int,
        max_steps_per_episode: int = 1000,
        warmup_steps: int = 0,
        checkpoint_dir: str | Path | None = None,
        save_every: int = 50,
        checkpoint_name: str = "dqn_agent.pt",
    ) -> None:
        if num_episodes <= 0:
            raise ValueError("num_episodes must be > 0")
        if max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be > 0")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if save_every <= 0:
            raise ValueError("save_every must be > 0")

        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.warmup_steps = warmup_steps
        self.save_every = save_every
        self.checkpoint_name = checkpoint_name

        self.global_step = 0
        self.history: list[TrainStats] = []

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, seed: int | None = None, verbose: bool = True) -> list[TrainStats]:
        """
        Run the full training loop.

        Args:
            seed: Optional seed for the first episode reset.
            verbose: If True, print episode summaries.

        Returns:
            List of TrainStats, one per episode.
        """
        for episode in range(1, self.num_episodes + 1):
            episode_seed = seed + episode - 1 if seed is not None else None
            stats = self.train_one_episode(episode=episode, seed=episode_seed)
            self.history.append(stats)

            if verbose:
                loss_str = (
                    f"{stats.average_loss:.4f}"
                    if stats.average_loss is not None
                    else "None"
                )
                print(
                    f"Episode {stats.episode}/{self.num_episodes} | "
                    f"reward={stats.episode_reward:.2f} | "
                    f"length={stats.episode_length} | "
                    f"avg_loss={loss_str} | "
                    f"epsilon={stats.epsilon:.4f}"
                )

            if self.checkpoint_dir is not None and episode % self.save_every == 0:
                checkpoint_path = self.checkpoint_dir / f"episode_{episode}_{self.checkpoint_name}"
                self.agent.save(str(checkpoint_path))

        if self.checkpoint_dir is not None:
            final_path = self.checkpoint_dir / f"final_{self.checkpoint_name}"
            self.agent.save(str(final_path))

        return self.history

    def train_one_episode(self, episode: int, seed: int | None = None) -> TrainStats:
        """
        Run one training episode.

        Args:
            episode: Current episode index.
            seed: Optional seed for env.reset().

        Returns:
            TrainStats for this episode.
        """
        state, _ = self.env.reset(seed=seed)

        episode_reward = 0.0
        losses: list[float] = []
        episode_length = 0

        for step in range(1, self.max_steps_per_episode + 1):
            action = self.agent.act(state, training=True)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.agent.store_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            if self.global_step >= self.warmup_steps:
                loss = self.agent.update()
                if loss is not None:
                    losses.append(loss)

            state = next_state
            episode_reward += float(reward)
            episode_length = step
            self.global_step += 1

            if done:
                break

        average_loss = float(np.mean(losses)) if losses else None

        return TrainStats(
            episode=episode,
            episode_reward=episode_reward,
            episode_length=episode_length,
            average_loss=average_loss,
            epsilon=self.agent.epsilon,
        )

    def evaluate(
        self,
        num_episodes: int = 5,
        max_steps_per_episode: int | None = None,
        seed: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate the agent without exploration.

        Args:
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Optional max steps override.
            seed: Optional seed for deterministic resets.

        Returns:
            Summary metrics dictionary.
        """
        if num_episodes <= 0:
            raise ValueError("num_episodes must be > 0")

        max_steps = max_steps_per_episode or self.max_steps_per_episode

        rewards: list[float] = []
        lengths: list[int] = []

        for episode in range(num_episodes):
            episode_seed = seed + episode if seed is not None else None
            state, _ = self.env.reset(seed=episode_seed)

            episode_reward = 0.0
            episode_length = 0

            for step in range(1, max_steps + 1):
                action = self.agent.act(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                state = next_state
                episode_reward += float(reward)
                episode_length = step

                if done:
                    break

            rewards.append(episode_reward)
            lengths.append(episode_length)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(lengths)),
        }

    def get_history_dicts(self) -> list[dict[str, float | int | None]]:
        """
        Convert training history into a list of dictionaries.
        """
        return [
            {
                "episode": stat.episode,
                "episode_reward": stat.episode_reward,
                "episode_length": stat.episode_length,
                "average_loss": stat.average_loss,
                "epsilon": stat.epsilon,
            }
            for stat in self.history
        ]