"""Reproducibility helper — seed all RNGs used by the pipeline."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Seed Python, NumPy and torch (CPU + CUDA) and make cuDNN deterministic so a run
    can be reproduced. Call at the start of each training fold.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic convolutions / disable autotuner picking different algorithms.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
