import random
import numpy as np
import torch
import pandas as pd


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_device() -> torch.device:
    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     device = torch.device('mps')    # for Apple Macbook GPUs
    print(f'using device: {device}')
    return device

    