import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.grpo_agent import GRPOAgent
from src.agents.ppo_agent import PPOAgent


def make_ppo_agent() -> PPOAgent:
    return PPOAgent(
        state_dim=8,
        action_dim=4,
        hidden_dims=(64, 64),
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        update_epochs=2,
        mini_batch_size=4,
        max_grad_norm=0.5,
        group_size=4,
        device="cpu",
    )


def make_grpo_agent() -> GRPOAgent:
    return GRPOAgent(
        state_dim=8,
        action_dim=4,
        hidden_dims=(64, 64),
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        update_epochs=2,
        mini_batch_size=4,
        max_grad_norm=0.5,
        group_size=4,
        device="cpu",
    )


def test_ppo_act_returns_valid_action():
    agent = make_ppo_agent()
    state = np.random.randn(8).astype(np.float32)
    action = agent.act(state, training=False)
    assert 0 <= action < 4


def test_grpo_act_returns_valid_action():
    agent = make_grpo_agent()
    state = np.random.randn(8).astype(np.float32)
    action = agent.act(state, training=False)
    assert 0 <= action < 4


def test_ppo_update_runs_and_clears_buffer():
    agent = make_ppo_agent()

    for i in range(8):
        state = np.random.randn(8).astype(np.float32)
        action, log_prob, value = agent.sample_action(state)
        agent.store_transition(
            state=state,
            action=action,
            reward=float(np.random.randn()),
            done=(i == 7),
            log_prob=log_prob,
            value=float(value),
        )

    loss = agent.update(last_state=None, last_done=True)
    assert isinstance(loss, float)
    assert len(agent.buffer) == 0
    assert agent.update_steps == 1


def test_grpo_update_runs_and_clears_buffer():
    agent = make_grpo_agent()

    for i in range(8):
        state = np.random.randn(8).astype(np.float32)
        action, log_prob, _ = agent.sample_action(state)
        agent.store_transition(
            state=state,
            action=action,
            reward=float(np.random.randn()),
            done=(i == 7),
            log_prob=log_prob,
            value=None,
        )

    loss = agent.update(last_state=None, last_done=True)
    assert isinstance(loss, float)
    assert len(agent.buffer) == 0
    assert agent.update_steps == 1


def test_ppo_invalid_state_dim():
    with pytest.raises(ValueError, match="state_dim must be > 0"):
        PPOAgent(state_dim=0, action_dim=4)


def test_grpo_invalid_action_dim():
    with pytest.raises(ValueError, match="action_dim must be > 0"):
        GRPOAgent(state_dim=8, action_dim=0)
