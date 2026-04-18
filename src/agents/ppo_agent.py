from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

from src.agents.base_agent import BaseAgent
from src.memory.trajectory_buffer import TrajectoryBuffer


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        dims = [state_dim, *hidden_dims]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(dims[-1], action_dim)
        self.value_head = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value


class PPOAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 4,
        mini_batch_size: int = 64,
        max_grad_norm: float = 0.5,
        group_size: int = 32,
        device: str = "cpu",
    ) -> None:
        super().__init__(device=device)

        if state_dim <= 0:
            raise ValueError("state_dim must be > 0")
        if action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if clip_eps <= 0:
            raise ValueError("clip_eps must be > 0")
        if entropy_coef < 0:
            raise ValueError("entropy_coef must be >= 0")
        if value_coef < 0:
            raise ValueError("value_coef must be >= 0")
        if update_epochs <= 0:
            raise ValueError("update_epochs must be > 0")
        if mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be > 0")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0")
        if group_size <= 0:
            raise ValueError("group_size must be > 0")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        # Keep this parameter for shared benchmark configs with GRPO.
        self.group_size = group_size

        self.network = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        )
        self.to_device(self.network)
        self.optimizer = Adam(self.network.parameters(), lr=learning_rate)

        self.buffer = TrajectoryBuffer()
        self.update_steps = 0

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def sample_action(self, state: np.ndarray) -> tuple[int, float, float]:
        state_t = self._state_tensor(state)
        self.eval_mode(self.network)
        with torch.no_grad():
            logits, value = self.network(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        self.train_mode(self.network)
        return int(action.item()), float(log_prob.item()), float(value.squeeze(-1).item())

    def act(self, state: np.ndarray, training: bool = True) -> int:
        state_t = self._state_tensor(state)
        self.eval_mode(self.network)
        with torch.no_grad():
            logits, _ = self.network(state_t)
            if training:
                action = Categorical(logits=logits).sample()
            else:
                action = torch.argmax(logits, dim=1)
        if training:
            self.train_mode(self.network)
        return int(action.item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.buffer.add(
            state=state,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
        )

    def _bootstrap_value(
        self,
        last_state: np.ndarray | None,
        last_done: bool,
    ) -> float:
        if last_done or last_state is None:
            return 0.0

        state_t = self._state_tensor(last_state)
        self.eval_mode(self.network)
        with torch.no_grad():
            _, value = self.network(state_t)
        return float(value.squeeze(-1).item())

    def _compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = last_value

        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        last_state: np.ndarray | None = None,
        last_done: bool = True,
    ) -> float | None:
        if len(self.buffer) == 0:
            return None

        trajectory = self.buffer.to_numpy()
        states = trajectory["states"]
        actions = trajectory["actions"]
        rewards = trajectory["rewards"]
        dones = trajectory["dones"]
        old_log_probs = trajectory["log_probs"]
        values = trajectory["values"]
        if values is None:
            raise ValueError("PPO requires value estimates in trajectory.")

        last_value = self._bootstrap_value(last_state=last_state, last_done=last_done)
        advantages, returns = self._compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            last_value=last_value,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        n_samples = states_t.size(0)
        losses: list[float] = []

        self.train_mode(self.network)
        for _ in range(self.update_epochs):
            indices = torch.randperm(n_samples, device=self.device)

            for start in range(0, n_samples, self.mini_batch_size):
                batch_idx = indices[start : start + self.mini_batch_size]

                logits, values_pred = self.network(states_t[batch_idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx])
                unclipped_obj = ratio * advantages_t[batch_idx]
                clipped_obj = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                ) * advantages_t[batch_idx]
                policy_loss = -torch.min(unclipped_obj, clipped_obj).mean()

                value_loss = nn.functional.mse_loss(
                    values_pred.squeeze(-1),
                    returns_t[batch_idx],
                )
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.item()))

        self.buffer.clear()
        self.update_steps += 1
        return float(np.mean(losses)) if losses else None

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_eps": self.clip_eps,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "update_epochs": self.update_epochs,
            "mini_batch_size": self.mini_batch_size,
            "max_grad_norm": self.max_grad_norm,
            "group_size": self.group_size,
            "update_steps": self.update_steps,
        }
        self.save_checkpoint(path, checkpoint)

    def load(self, path: str | Path) -> None:
        checkpoint = self.load_checkpoint(path)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_steps = int(checkpoint["update_steps"])
