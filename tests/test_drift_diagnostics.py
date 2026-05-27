import numpy as np

from driftscale.eval.drift import (
    demand_distribution_diagnostics,
    drift_diagnostic_passed,
    ks_statistic,
    standardized_mean_difference,
)


def test_demand_distribution_diagnostic_marks_measurable_shift() -> None:
    baseline = np.array([1.0, 1.1, 0.9, 1.0, 1.05])
    shifted = np.array([2.0, 2.1, 1.9, 2.0, 2.05])

    diagnostics = demand_distribution_diagnostics(
        [baseline, shifted],
        checkpoint_ids=[1, 25],
        mapping="linear",
        curriculum_source="azure",
        vm_selection_strategy="per_checkpoint_dense",
        selected_vm_count=2,
        min_drift_smd=0.5,
        min_drift_ks=0.3,
    )

    assert not diagnostics.loc[0, "measurable_shift"]
    assert diagnostics.loc[1, "measurable_shift"]
    assert diagnostics.loc[1, "mean"] > diagnostics.loc[0, "mean"]
    assert diagnostics.loc[1, "ks_vs_checkpoint_1"] == 1.0
    assert drift_diagnostic_passed(diagnostics)


def test_pairwise_drift_statistics_are_zero_for_identical_demands() -> None:
    demand = np.array([1.0, 1.2, 1.4, 1.6])

    assert standardized_mean_difference(demand, demand) == 0.0
    assert ks_statistic(demand, demand) == 0.0
