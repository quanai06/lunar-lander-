import numpy as np
import pytest



import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.replay_buffer import ReplayBuffer


def test_replay_buffer_init():
    buffer = ReplayBuffer(capacity=100)

    assert buffer.capacity == 100
    assert len(buffer) == 0


def test_replay_buffer_invalid_capacity():
    with pytest.raises(ValueError, match="capacity must be > 0"):
        ReplayBuffer(capacity=0)


def test_replay_buffer_add_one_transition():
    buffer = ReplayBuffer(capacity=10)

    state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    next_state = np.array([1.1, 2.1, 3.1], dtype=np.float32)

    buffer.add(
        state=state,
        action=1,
        reward=0.5,
        next_state=next_state,
        done=False,
    )

    assert len(buffer) == 1


def test_replay_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=3)

    for i in range(5):
        state = np.array([i, i + 1], dtype=np.float32)
        next_state = np.array([i + 0.1, i + 1.1], dtype=np.float32)

        buffer.add(
            state=state,
            action=i % 2,
            reward=float(i),
            next_state=next_state,
            done=False,
        )

    assert len(buffer) == 3


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(capacity=10)

    for i in range(5):
        state = np.array([i, i + 1, i + 2], dtype=np.float32)
        next_state = np.array([i + 0.5, i + 1.5, i + 2.5], dtype=np.float32)

        buffer.add(
            state=state,
            action=i % 4,
            reward=float(i),
            next_state=next_state,
            done=(i % 2 == 0),
        )

    states, actions, rewards, next_states, dones = buffer.sample(batch_size=4)

    assert states.shape == (4, 3)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 3)
    assert dones.shape == (4,)


def test_replay_buffer_sample_dtypes():
    buffer = ReplayBuffer(capacity=10)

    for i in range(4):
        state = np.array([i, i + 1], dtype=np.float32)
        next_state = np.array([i + 0.1, i + 1.1], dtype=np.float32)

        buffer.add(
            state=state,
            action=i,
            reward=float(i),
            next_state=next_state,
            done=(i % 2 == 0),
        )

    states, actions, rewards, next_states, dones = buffer.sample(batch_size=4)

    assert states.dtype == np.float32
    assert actions.dtype == np.int64
    assert rewards.dtype == np.float32
    assert next_states.dtype == np.float32
    assert dones.dtype == np.float32


def test_replay_buffer_sample_too_large():
    buffer = ReplayBuffer(capacity=10)

    state = np.array([1.0, 2.0], dtype=np.float32)
    next_state = np.array([1.5, 2.5], dtype=np.float32)

    buffer.add(
        state=state,
        action=0,
        reward=1.0,
        next_state=next_state,
        done=False,
    )

    with pytest.raises(ValueError, match="Not enough samples in buffer"):
        buffer.sample(batch_size=2)


def test_replay_buffer_invalid_batch_size():
    buffer = ReplayBuffer(capacity=10)

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        buffer.sample(batch_size=0)


def test_replay_buffer_is_ready():
    buffer = ReplayBuffer(capacity=10)

    for i in range(3):
        state = np.array([i, i + 1], dtype=np.float32)
        next_state = np.array([i + 0.1, i + 1.1], dtype=np.float32)
        buffer.add(
            state=state,
            action=0,
            reward=1.0,
            next_state=next_state,
            done=False,
        )

    assert buffer.is_ready(3) is True
    assert buffer.is_ready(4) is False


def test_replay_buffer_clear():
    buffer = ReplayBuffer(capacity=10)

    for i in range(3):
        state = np.array([i, i + 1], dtype=np.float32)
        next_state = np.array([i + 0.1, i + 1.1], dtype=np.float32)
        buffer.add(
            state=state,
            action=0,
            reward=1.0,
            next_state=next_state,
            done=False,
        )

    assert len(buffer) == 3

    buffer.clear()

    assert len(buffer) == 0