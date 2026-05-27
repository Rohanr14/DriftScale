"""Demand drift diagnostics for checkpoint curricula."""

from __future__ import annotations

import numpy as np
import pandas as pd


def demand_distribution_diagnostics(
    demands: list[np.ndarray],
    *,
    checkpoint_ids: list[int],
    mapping: str,
    curriculum_source: str,
    vm_selection_strategy: str,
    selected_vm_count: int,
    min_drift_smd: float,
    min_drift_ks: float,
) -> pd.DataFrame:
    """Summarize per-checkpoint demand distributions and drift from checkpoint 1."""
    if len(demands) != len(checkpoint_ids):
        raise ValueError("demands must align with checkpoint_ids")

    baseline = np.asarray(demands[0], dtype=np.float64)
    rows = []
    for stage, (checkpoint_id, demand) in enumerate(
        zip(checkpoint_ids, demands, strict=True),
        start=1,
    ):
        demand = np.asarray(demand, dtype=np.float64)
        smd = np.nan
        ks = np.nan
        measurable = False
        if stage > 1:
            smd = standardized_mean_difference(baseline, demand)
            ks = ks_statistic(baseline, demand)
            measurable = abs(smd) >= min_drift_smd or ks >= min_drift_ks
        rows.append(
            {
                "curriculum_source": curriculum_source,
                "mapping": mapping,
                "vm_selection_strategy": vm_selection_strategy,
                "selected_vm_count": selected_vm_count,
                "stage": stage,
                "checkpoint": checkpoint_id,
                "mean": float(np.mean(demand)),
                "std": float(np.std(demand, ddof=1)),
                "p95": float(np.percentile(demand, 95)),
                "min": float(np.min(demand)),
                "max": float(np.max(demand)),
                "smd_vs_checkpoint_1": float(smd),
                "ks_vs_checkpoint_1": float(ks),
                "measurable_shift": bool(measurable),
            }
        )
    return pd.DataFrame(rows)


def standardized_mean_difference(baseline: np.ndarray, current: np.ndarray) -> float:
    """Return the difference in means normalized by pooled standard deviation."""
    pooled_std = np.sqrt((np.var(baseline, ddof=1) + np.var(current, ddof=1)) / 2.0)
    if pooled_std <= 0.0:
        return 0.0
    return float((np.mean(current) - np.mean(baseline)) / pooled_std)


def ks_statistic(baseline: np.ndarray, current: np.ndarray) -> float:
    """Return the two-sample Kolmogorov-Smirnov statistic without scipy."""
    left = np.sort(np.asarray(baseline, dtype=np.float64))
    right = np.sort(np.asarray(current, dtype=np.float64))
    values = np.sort(np.concatenate([left, right]))
    left_cdf = np.searchsorted(left, values, side="right") / max(len(left), 1)
    right_cdf = np.searchsorted(right, values, side="right") / max(len(right), 1)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def drift_diagnostic_passed(diagnostics: pd.DataFrame) -> bool:
    """Return true when every post-baseline checkpoint has measurable shift."""
    shifted = diagnostics[diagnostics["stage"] > 1]
    return bool(not shifted.empty and shifted["measurable_shift"].all())
