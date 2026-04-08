from __future__ import annotations
 
import numpy as np 

from typing import Any
from collections import deque
from dataclasses import dataclass
import random

@dataclass
class Transition :
    state :np.ndarray
    action :int 
    reward : float
    next_state:np.ndarray
    done:bool 

class ReplayBuffer:
    def __init__(self, capacity :int)-> None :
        """
        Replay buffer for storing agent transitions.

        Args:
        """

        if capacity <=0:
            raise ValueError("capacity must be > 0 ")
        
        self.capacity=capacity
        self.buffer: deque[Transition]=deque(maxlen=capacity)
    def add(
        self,
        state :np.ndarray,
        action :int,
        reward :float,
        next_state:np.ndarray,
        done :bool
    )-> None:
        """Add one transition to the replay buffer"""

        transition=Transition(
            state=np.asarray(state,dtype=np.float32),
            action = int(action),
            reward=float(reward),
            next_state= np.asarray(next_state,dtype=np.float32),
            done = bool(done),
        )
        self.buffer.append(transition)
    def sample (self, batch_size :int ) -> tuple[np.ndarray,...]:
        """
        Randomly sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            A tuple of:
                states, actions, rewards, next_states, dones
        """
        if batch_size <=0 :
            raise ValueError(
                f"Not enough samples in buffer : requested{batch_size},available{len(batch_size)}"
            )
        batch=random.sample(self.buffer, batch_size)

        states= np.stack([transition.state for transition in batch])
        actions = np.array([transition.action for transition in batch],dtype=np.int64)
        rewards= np.array([transition.reward for transition in batch],dtype=np.float32)
        next_states=np.stack([transition.next_state for transition in batch])
        dones = np.array([transition.done for transition in batch],dtype=np.float32)

        return states, actions, rewards, next_states, dones 
    
    def __len__(self)-> int :
        return len(self.buffer)
    def is_ready(self, batch_size :int)-> bool :
        if batch_size<=0 :
            raise ValueError(" batch size must be > 0 ")
        return len(self.buffer) >= batch_size
    
    def clear(self)-> None:
        self.buffer.clear()