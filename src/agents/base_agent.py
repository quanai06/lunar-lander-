from __future__ import annotations

from pathlib import Path
from typing import Any

import torch 
from torch import nn

class BaseAgent:
    def __init__(self, device :str | torch.device="cpu")-> None:

        self.device=torch.device(device)

    def to_device(self,*modules:nn.Module)-> None:

        for module in modules :
            module.to(self.device)

    def train_mode(self, *modules:nn.Module) -> None :
        for module in modules:
            module.train()

    def eval_mode(self, *modules :nn.Module) -> None:

        for module in modules:
            module.eval()
    def save_checkpoint(self,path:str |Path, checkpoint : dict[str,Any]) -> None:

        path= Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path :str | Path)-> dict[str,Any]:

        path=Path(path)

        if not path.exists():
            raise FileNotFoundError(f'checkpoint not found : {path}')
        
        checkpoint= torch.load(path, map_location=self.device)
        return checkpoint
    def act(self,*arg,**kwargs):
        raise NotImplementedError('Subclasses must implement act().')
    
    def update(self,*arg,**kwargs):
        raise NotImplementedError("subclasses must implement update().")
    
