import torch
import numpy as np
import random

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False