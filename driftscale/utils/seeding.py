"""Reproducibility helpers."""

from __future__ import annotations

import importlib
import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and (when present) PyTorch RNGs from one integer.

    Torch is imported lazily so the dev extra (no torch) keeps working without it.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
