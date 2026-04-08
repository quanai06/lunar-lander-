from __future__ import annotations

import os 
import random 

import numpy as np  
import torch 

def set_seed( seed:int , deterministic : bool =True) -> None :
    """ set seed for numpy python and pytorch """
    if seed <0 :
        raise ValueError("seed must be >=0")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed=seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"]= str(seed)

    if deterministic :

        # Make cuDNN more deterministic, sometimes slower 
        torch.backends.cudnn.deterministic =True
        torch.backends.cudnn.benchmark= False

        try :
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic =False
        torch.backends.cudnn.benchmark=True

def seed_env(env,seed:int) -> None:
    """seed for gymnasium"""

    if hasattr(env,'action_space') and env.action_space is not None:
        env.action_space.seed(seed)
    if hasattr(env,'observation_space') and env.observation_space is not None :
        env.observation_space.seed(seed)