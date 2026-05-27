"""Reward calculation for the DriftScale simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights and clipping settings for the autoscaling objective."""

    cost_weight: float = 1.0
    cost_per_task_step: float = 0.05
    slo_weight: float = 50.0
    action_weight: float = 0.5
    overprovision_weight: float = 2.0
    max_penalty_ratio: float = 1.0
    eps: float = 1e-8


@dataclass(frozen=True)
class RewardBreakdown:
    """Named components for testing, metrics, and debugging."""

    resource_cost: float
    slo_penalty: float
    scaling_action_penalty: float
    overprovision_penalty: float
    total_penalty: float
    reward: float
    slo_violation: bool
    utilization: float
    overprovision_ratio: float


def calculate_reward(
    *,
    demand: float,
    capacity: float,
    old_task_count: int,
    new_task_count: int,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Calculate the clipped negative-cost reward from the project bible."""
    cfg = config or RewardConfig()
    demand = max(float(demand), 0.0)
    capacity = max(float(capacity), 0.0)

    deficit = max(0.0, demand - capacity)
    spare_capacity = max(0.0, capacity - demand)

    slo_ratio = min(deficit / max(demand, cfg.eps), cfg.max_penalty_ratio)
    overprovision_ratio = min(spare_capacity / max(capacity, cfg.eps), cfg.max_penalty_ratio)
    utilization = min(demand / max(capacity, cfg.eps), 1.0)

    resource_cost = cfg.cost_weight * new_task_count * cfg.cost_per_task_step
    slo_penalty = cfg.slo_weight * slo_ratio
    scaling_action_penalty = cfg.action_weight * abs(new_task_count - old_task_count)
    overprovision_penalty = cfg.overprovision_weight * overprovision_ratio
    total_penalty = resource_cost + slo_penalty + scaling_action_penalty + overprovision_penalty

    return RewardBreakdown(
        resource_cost=resource_cost,
        slo_penalty=slo_penalty,
        scaling_action_penalty=scaling_action_penalty,
        overprovision_penalty=overprovision_penalty,
        total_penalty=total_penalty,
        reward=-total_penalty,
        slo_violation=deficit > 0.0,
        utilization=utilization,
        overprovision_ratio=overprovision_ratio,
    )
