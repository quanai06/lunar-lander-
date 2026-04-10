import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.dqn_agent import DQNAgent


def make_agent() -> DQNAgent:
    return DQNAgent(
        state_dim=8,
        action_dim=4,
        buffer_capacity=100,
        hidden_dims=(64, 64),
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=4,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9,
        target_update_freq=2,
        device="cpu",
    )


def make_transition():
    state = np.random.randn(8).astype(np.float32)
    action = int(np.random.randint(0, 4))
    reward = float(np.random.randn())
    next_state = np.random.randn(8).astype(np.float32)
    done = bool(np.random.rand() > 0.5)

    return state, action, reward, next_state, done


def test_dqn_agent_init():
    agent = make_agent()

    assert agent.state_dim == 8
    assert agent.action_dim == 4
    assert agent.batch_size == 4
    assert agent.gamma == 0.99
    assert agent.epsilon == 1.0
    assert agent.target_update_freq == 2
    assert agent.update_steps == 0


def test_dqn_agent_invalid_state_dim():
    with pytest.raises(ValueError, match="state_dim must be > 0"):
        DQNAgent(state_dim=0, action_dim=4)


def test_dqn_agent_invalid_action_dim():
    with pytest.raises(ValueError, match="action_dim must be > 0"):
        DQNAgent(state_dim=8, action_dim=0)


def test_dqn_agent_act_returns_valid_action():
    agent = make_agent()
    state = np.random.randn(8).astype(np.float32)

    action = agent.act(state, training=True)

    assert isinstance(action, int)
    assert 0 <= action < 4


def test_dqn_agent_act_greedy_mode_returns_valid_action():
    agent = make_agent()
    state = np.random.randn(8).astype(np.float32)

    action = agent.act(state, training=False)

    assert isinstance(action, int)
    assert 0 <= action < 4


def test_store_transition_increases_buffer_length():
    agent = make_agent()

    assert len(agent.replay_buffer) == 0

    state, action, reward, next_state, done = make_transition()
    agent.store_transition(state, action, reward, next_state, done)

    assert len(agent.replay_buffer) == 1


def test_update_returns_none_when_buffer_not_ready():
    agent = make_agent()

    for _ in range(3):  # batch_size = 4, so not enough yet
        state, action, reward, next_state, done = make_transition()
        agent.store_transition(state, action, reward, next_state, done)

    loss = agent.update()

    assert loss is None


def test_update_returns_float_when_buffer_ready():
    agent = make_agent()

    for _ in range(4):  # enough for one batch
        state, action, reward, next_state, done = make_transition()
        agent.store_transition(state, action, reward, next_state, done)

    loss = agent.update()

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_update_increments_update_steps():
    agent = make_agent()

    for _ in range(4):
        state, action, reward, next_state, done = make_transition()
        agent.store_transition(state, action, reward, next_state, done)

    assert agent.update_steps == 0

    agent.update()

    assert agent.update_steps == 1


def test_decay_epsilon_eventually_reaches_epsilon_end():
    agent = make_agent()

    for _ in range(100):
        agent.decay_epsilon()

    assert agent.epsilon == agent.epsilon_end


def test_update_target_network_copies_weights():
    agent = make_agent()

    with torch.no_grad():
        for param in agent.q_network.parameters():
            param.add_(1.0)

    different_before = False
    for q_param, target_param in zip(agent.q_network.parameters(), agent.target_network.parameters()):
        if not torch.allclose(q_param, target_param):
            different_before = True
            break

    assert different_before is True

    agent.update_target_network()

    for q_param, target_param in zip(agent.q_network.parameters(), agent.target_network.parameters()):
        assert torch.allclose(q_param, target_param)


def test_target_network_updates_on_frequency():
    agent = make_agent()

    for _ in range(8):
        state, action, reward, next_state, done = make_transition()
        agent.store_transition(state, action, reward, next_state, done)

    initial_target_params = [
        param.clone().detach() for param in agent.target_network.parameters()
    ]

    agent.update()  # update_steps = 1, no sync yet
    target_params_after_first_update = [
        param.clone().detach() for param in agent.target_network.parameters()
    ]

    same_after_first = all(
        torch.allclose(p0, p1)
        for p0, p1 in zip(initial_target_params, target_params_after_first_update)
    )
    assert same_after_first is True

    agent.update()  # update_steps = 2, sync should happen
    target_params_after_second_update = [
        param.clone().detach() for param in agent.target_network.parameters()
    ]

    changed_after_second = any(
        not torch.allclose(p0, p1)
        for p0, p1 in zip(initial_target_params, target_params_after_second_update)
    )
    assert changed_after_second is True