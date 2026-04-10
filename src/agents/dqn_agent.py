from __future__ import annotations

from copy import deepcopy

import numpy as np 
import torch 
from torch import nn 
from torch.optim import Adam

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.base_agent import BaseAgent
from src.memory.replay_buffer import ReplayBuffer
from src.models.mlp import MLP

class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_dim:int,
        action_dim:int,
        buffer_capacity :int =100000,
        hidden_dims :tuple[int,...]=(128,128),
        learning_rate :float=1e-3,
        gamma :float=0.99,
        batch_size:int =64,
        epsilon_start:float =1.0,
        epsilon_end: float = 0.05,
        epsilon_decay=0.995,
        target_update_freq:int=100,
        device :str ="cpu"
    )-> None:
        super().__init__(device=device)

        if state_dim <= 0:
            raise ValueError("state_dim must be > 0")
        if action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be > 0")
        if target_update_freq <= 0:
            raise ValueError("target_update_freq must be > 0")
        if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_network=MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims
        )
        self.target_network=MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims
        )

        self.to_device(self.q_network, self.target_network)

        self.target_network.load_state_dict(deepcopy(self.q_network.state_dict()))
        self.eval_mode(self.target_network)

        self.optimizer =Adam(self.q_network.parameters(),lr=learning_rate)
        self.loss_fn=nn.MSELoss()

        self.replay_buffer =ReplayBuffer(capacity=buffer_capacity)

        self.update_steps =0

    def act(self, state:np.ndarray,training : bool = True) -> int:
        if training and np.random.rand() <self.epsilon :
            return int(np.random.randint(self.action_dim))
        state_tensor =torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.eval_mode(self.q_network)
        with torch.no_grad():
            q_values= self.q_network(state_tensor)
            action= int(torch.argmax(q_values, dim=1).item())

        if training :
            self.train_mode(self.q_network)

        return action 
    def store_transition(
        self,
        state:np.ndarray,
        action:int,
        reward :float,
        next_state:np.ndarray,
        done:bool,
    )-> None:
        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )

    def update(self) -> float |None :
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.train_mode(self.q_network)
        self.eval_mode(self.target_network)

        current_q_values=self.q_network(states_t).gather(1,actions_t)

        with torch.no_grad():
            max_next_q_values= self.target_network(next_states_t).max(dim=1,keepdim=True)[0]
            target_q_values= rewards_t+self.gamma* max_next_q_values*(1.0-dones_t)

        loss =self.loss_fn(current_q_values,target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps+=1 

        if self.update_steps% self.target_update_freq==0:
            self.update_target_network()

        self.decay_epsilon()

        return float(loss.item())
    def update_target_network(self)-> None :
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self)-> None:

        self.epsilon =max(self.epsilon_end,self.epsilon* self.epsilon_decay)

    def save(self, path: str) -> None:
        """
        Save agent checkpoint.
        """
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_steps": self.update_steps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
        }
        self.save_checkpoint(path, checkpoint)

    def load(self, path: str) -> None:
        """
        Load agent checkpoint.
        """
        checkpoint = self.load_checkpoint(path)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.epsilon = float(checkpoint["epsilon"])
        self.update_steps = int(checkpoint["update_steps"])