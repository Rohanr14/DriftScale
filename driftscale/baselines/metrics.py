"""Shared baseline evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AutoscalingMetrics:
    """Aggregate metrics from one policy rollout."""

    policy_name: str
    total_reward: float
    total_resource_cost: float
    total_scaling_cost: float
    slo_violation_rate: float
    scale_action_count: int
    mean_task_count: float
    mean_overprovision_ratio: float
    final_task_count: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "policy": self.policy_name,
            "total_reward": self.total_reward,
            "total_resource_cost": self.total_resource_cost,
            "total_scaling_cost": self.total_scaling_cost,
            "slo_violation_rate": self.slo_violation_rate,
            "scale_action_count": self.scale_action_count,
            "mean_task_count": self.mean_task_count,
            "mean_overprovision_ratio": self.mean_overprovision_ratio,
            "final_task_count": self.final_task_count,
        }


def mean_overprovision_ratio(demand: np.ndarray, capacity: np.ndarray) -> float:
    spare = np.maximum(capacity - demand, 0.0)
    ratios = spare / np.maximum(capacity, 1e-8)
    return float(np.mean(ratios))

