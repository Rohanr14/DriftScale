import numpy as np
import pytest

from driftscale.traces.regimes import PHASE1_REGIMES
from driftscale.traces.synthetic import generate_synthetic_episode


def test_synthetic_generation_is_deterministic() -> None:
    first = generate_synthetic_episode(regime="bursty", length=96, seed=11)
    second = generate_synthetic_episode(regime="bursty", length=96, seed=11)

    np.testing.assert_allclose(first.demand, second.demand)


@pytest.mark.parametrize("regime", PHASE1_REGIMES)
def test_synthetic_regimes_are_positive(regime: str) -> None:
    episode = generate_synthetic_episode(regime=regime, length=96, seed=3)

    assert episode.demand.shape == (96,)
    assert np.all(episode.demand > 0.0)
    assert episode.metadata["p95_demand"] >= episode.metadata["mean_demand"]

