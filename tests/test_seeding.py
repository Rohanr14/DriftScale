import importlib

import numpy as np

from driftscale.utils.seeding import seed_everything


def test_seed_everything_makes_numpy_deterministic() -> None:
    seed_everything(123)
    first = np.random.random(4)
    seed_everything(123)
    second = np.random.random(4)
    assert np.array_equal(first, second)


def test_seed_everything_seeds_torch_when_available() -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:  # train extras absent — torch test is skipped silently
        return
    seed_everything(7)
    a = torch.rand(3)
    seed_everything(7)
    b = torch.rand(3)
    assert torch.equal(a, b)
