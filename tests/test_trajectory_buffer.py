import numpy as np
import pytest

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.trajectory_buffer import TrajectoryBuffer


def test_trajectory_buffer_add_and_len():
    buffer = TrajectoryBuffer()

    assert len(buffer) == 0

    buffer.add(
        state=np.array([1.0, 2.0], dtype=np.float32),
        action=1,
        reward=0.5,
        done=False,
        log_prob=-0.3,
        value=0.8,
    )

    assert len(buffer) == 1


def test_trajectory_buffer_to_numpy_with_values():
    buffer = TrajectoryBuffer()
    for i in range(3):
        buffer.add(
            state=np.array([i, i + 1], dtype=np.float32),
            action=i % 2,
            reward=float(i),
            done=False,
            log_prob=-0.1 * i,
            value=0.2 * i,
        )

    arrays = buffer.to_numpy()

    assert arrays["states"].shape == (3, 2)
    assert arrays["actions"].shape == (3,)
    assert arrays["rewards"].shape == (3,)
    assert arrays["dones"].shape == (3,)
    assert arrays["log_probs"].shape == (3,)
    assert arrays["values"] is not None
    assert arrays["values"].shape == (3,)


def test_trajectory_buffer_to_numpy_without_values():
    buffer = TrajectoryBuffer()
    for i in range(2):
        buffer.add(
            state=np.array([i, i + 1], dtype=np.float32),
            action=i,
            reward=1.0,
            done=(i == 1),
            log_prob=-0.2,
            value=None,
        )

    arrays = buffer.to_numpy()
    assert arrays["values"] is None


def test_trajectory_buffer_empty_to_numpy_raises():
    buffer = TrajectoryBuffer()
    with pytest.raises(ValueError, match="TrajectoryBuffer is empty"):
        buffer.to_numpy()
