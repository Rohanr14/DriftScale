"""Statistical-significance and bootstrap helpers for forgetting metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile bootstrap mean and confidence interval."""

    mean: float
    ci_low: float
    ci_high: float
    n: int
    resamples: int
    alpha: float

    def as_dict(self, prefix: str = "") -> dict[str, float | int]:
        return {
            f"{prefix}mean": self.mean,
            f"{prefix}ci_low": self.ci_low,
            f"{prefix}ci_high": self.ci_high,
            f"{prefix}n": self.n,
            f"{prefix}resamples": self.resamples,
            f"{prefix}alpha": self.alpha,
        }


@dataclass(frozen=True)
class PairedTestResult:
    """Two-sided Wilcoxon signed-rank test outcome with a sign-aware effect direction."""

    statistic: float
    p_value: float
    n: int
    median_difference: float

    def as_dict(self, prefix: str = "") -> dict[str, float | int]:
        return {
            f"{prefix}wilcoxon_stat": self.statistic,
            f"{prefix}p_value": self.p_value,
            f"{prefix}n": self.n,
            f"{prefix}median_difference": self.median_difference,
        }


def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    *,
    resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for the sample mean.

    Returns a degenerate CI (low == high == mean) when the input has length < 2 or zero
    variance, so callers never hit an empty-percentile crash.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    n = int(array.size)
    if n == 0:
        return BootstrapCI(mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                           n=0, resamples=resamples, alpha=alpha)
    mean = float(np.mean(array))
    if n < 2 or float(np.std(array)) == 0.0:
        return BootstrapCI(mean=mean, ci_low=mean, ci_high=mean, n=n,
                           resamples=resamples, alpha=alpha)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(resamples, n))
    means = array[indices].mean(axis=1)
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    ci_low, ci_high = np.percentile(means, [lower_q, upper_q])
    return BootstrapCI(
        mean=mean,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n=n,
        resamples=resamples,
        alpha=alpha,
    )


def paired_wilcoxon(
    baseline: np.ndarray | list[float],
    treatment: np.ndarray | list[float],
) -> PairedTestResult:
    """Two-sided Wilcoxon signed-rank test on per-seed paired metrics.

    Returns (treatment - baseline) median for sign-aware interpretation. NaN p-value is
    returned when n < 2 (the test is undefined) or every difference is exactly zero.
    """
    baseline_array = np.asarray(baseline, dtype=np.float64).ravel()
    treatment_array = np.asarray(treatment, dtype=np.float64).ravel()
    if baseline_array.shape != treatment_array.shape:
        raise ValueError("paired arrays must have equal shape")
    n = int(baseline_array.size)
    differences = treatment_array - baseline_array
    median_diff = float(np.median(differences)) if n > 0 else float("nan")
    if n < 2 or np.all(differences == 0.0):
        return PairedTestResult(
            statistic=float("nan"),
            p_value=float("nan"),
            n=n,
            median_difference=median_diff,
        )
    result = stats.wilcoxon(
        treatment_array,
        baseline_array,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return PairedTestResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n=n,
        median_difference=median_diff,
    )
