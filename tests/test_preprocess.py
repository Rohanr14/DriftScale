import numpy as np
import pandas as pd

from driftscale.traces.preprocess import compute_demand_signal


def test_mapping_variants_are_vectorized_and_mathematically_distinct() -> None:
    cpu = pd.DataFrame(
        [
            [0.20, 0.50, 0.95],
            [0.70, 0.10, 0.20],
            [0.40, 0.92, 0.30],
        ],
        index=pd.date_range("2017-01-01", periods=3, freq="5min", tz="UTC"),
    )

    linear = compute_demand_signal(cpu, variant="linear", scale_factor=2.0, alpha=3.0)
    convex = compute_demand_signal(cpu, variant="convex", scale_factor=2.0, alpha=3.0)
    threshold = compute_demand_signal(cpu, variant="threshold", scale_factor=2.0, alpha=3.0)

    np.testing.assert_allclose(linear, 2.0 * cpu.to_numpy().sum(axis=1))
    np.testing.assert_allclose(convex, 2.0 * np.power(cpu.to_numpy(), 1.5).sum(axis=1))
    np.testing.assert_allclose(
        threshold,
        (2.0 * cpu.to_numpy().sum(axis=1)) + np.array([3.0, 0.0, 3.0]),
    )
    assert not np.allclose(linear, convex)
    assert not np.allclose(linear, threshold)
    assert not np.allclose(convex, threshold)

