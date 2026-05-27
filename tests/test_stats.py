import math

import numpy as np
import pytest

from driftscale.eval.stats import bootstrap_mean_ci, paired_wilcoxon


def test_bootstrap_ci_brackets_the_mean() -> None:
    rng = np.random.default_rng(123)
    values = rng.normal(loc=5.0, scale=1.0, size=200)
    ci = bootstrap_mean_ci(values, resamples=2000, seed=42)
    assert ci.ci_low < ci.mean < ci.ci_high
    assert ci.ci_low < 5.0 < ci.ci_high
    assert ci.n == 200


def test_bootstrap_ci_is_deterministic_under_seed() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = bootstrap_mean_ci(values, resamples=500, seed=7)
    b = bootstrap_mean_ci(values, resamples=500, seed=7)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_bootstrap_ci_handles_constant_and_singleton_inputs() -> None:
    constant = bootstrap_mean_ci([3.0, 3.0, 3.0])
    assert constant.ci_low == constant.ci_high == 3.0
    singleton = bootstrap_mean_ci([7.0])
    assert singleton.ci_low == singleton.ci_high == 7.0


def test_bootstrap_ci_empty_returns_nan() -> None:
    empty = bootstrap_mean_ci([])
    assert math.isnan(empty.mean)
    assert empty.n == 0


def test_paired_wilcoxon_detects_consistent_improvement() -> None:
    baseline = np.array([-10.0, -12.0, -8.0, -11.0, -9.0])
    treatment = baseline + 5.0
    result = paired_wilcoxon(baseline, treatment)
    assert result.p_value < 0.1
    assert result.median_difference == pytest.approx(5.0)
    assert result.n == 5


def test_paired_wilcoxon_identical_inputs_yield_nan_pvalue() -> None:
    same = np.array([1.0, 2.0, 3.0])
    result = paired_wilcoxon(same, same)
    assert math.isnan(result.p_value)
    assert result.median_difference == 0.0


def test_paired_wilcoxon_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        paired_wilcoxon([1.0, 2.0], [1.0, 2.0, 3.0])
