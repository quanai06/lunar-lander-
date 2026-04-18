from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryTransition:
    state: np.ndarray
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float | None = None


class TrajectoryBuffer:
    def __init__(self) -> None:
        self.transitions: list[TrajectoryTransition] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float | None = None,
    ) -> None:
        self.transitions.append(
            TrajectoryTransition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                done=bool(done),
                log_prob=float(log_prob),
                value=float(value) if value is not None else None,
            )
        )

    def to_numpy(self) -> dict[str, np.ndarray]:
        if len(self.transitions) == 0:
            raise ValueError("TrajectoryBuffer is empty.")

        states = np.stack([t.state for t in self.transitions])
        actions = np.array([t.action for t in self.transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
        dones = np.array([t.done for t in self.transitions], dtype=np.float32)
        log_probs = np.array([t.log_prob for t in self.transitions], dtype=np.float32)

        has_values = any(t.value is not None for t in self.transitions)
        values = None
        if has_values:
            values = np.array(
                [0.0 if t.value is None else t.value for t in self.transitions],
                dtype=np.float32,
            )

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "log_probs": log_probs,
            "values": values,
        }

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)
