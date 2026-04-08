from __future__ import annotations

from typing import Sequence

import torch 
from torch import nn 


class MLP(nn.Module):
    def __init__(
        self,
        input_dim :int,
        output_dim :int,
        hidden_dims:Sequence[int]=(128,128),
    )-> None:
        """
        Simple multi-layer perceptron.

        Args:
            input_dim: Number of input features.
            output_dim: Number of output features.
            hidden_dims: Sizes of hidden layers.
        """


        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("all hidden_dims must be > 0")
        
        layers :list[nn.Module]=[]
        layer_dims = [input_dim, *hidden_dims,output_dim]

        for i in range (len(layer_dims)-1):
            in_features= layer_dims[i]
            out_features=layer_dims[i+1]

            layers.append(nn.Linear(in_features, out_features))

            if i< len(layer_dims)-2 :
                layers.append(nn.ReLU())

        self.network= nn.Sequential(*layers)

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)