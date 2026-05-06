import os
import random
import numpy as np
import torch
from transformers import set_seed
from datetime import datetime

def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)

def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))

def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))

def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return v.strip().lower() in ["1", "true", "yes", "y"]

def seed_everything(seed: int = 42):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_formatted_datetime_string(format_code="%Y-%m-%d_%H-%M-%S"):
    now = datetime.now()
    return now.strftime(format_code)