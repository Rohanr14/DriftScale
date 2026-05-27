"""Static autoscaling baselines."""

from __future__ import annotations

import math

import numpy as np

from driftscale.baselines.metrics import AutoscalingMetrics, mean_overprovision_ratio
from driftscale.envs.reward import RewardConfig, calculate_reward


def static_task_count(
    demand: np.ndarray | list[float],
    *,
    quantile: float,
    capacity_per_task: float = 1.0,
    min_tasks: int = 1,
    max_tasks: int = 20,
) -> int:
    """Choose a fixed task count from a demand quantile."""
    demand_array = np.asarray(demand, dtype=np.float32)
    raw_count = math.ceil(float(np.quantile(demand_array, quantile)) / capacity_per_task)
    return int(np.clip(raw_count, min_tasks, max_tasks))


def evaluate_static_policy(
    demand: np.ndarray | list[float],
    *,
    task_count: int,
    policy_name: str,
    capacity_per_task: float = 1.0,
    reward_config: RewardConfig | None = None,
) -> AutoscalingMetrics:
    """Evaluate a fixed task-count policy over a demand sequence."""
    reward_cfg = reward_config or RewardConfig()
    demand_array = np.asarray(demand, dtype=np.float32)
    capacity = np.full_like(demand_array, task_count * capacity_per_task)
    rewards = []
    resource_costs = []
    slo_violations = []

    for demand_t, capacity_t in zip(demand_array, capacity, strict=True):
        breakdown = calculate_reward(
            demand=float(demand_t),
            capacity=float(capacity_t),
            old_task_count=task_count,
            new_task_count=task_count,
            config=reward_cfg,
        )
        rewards.append(breakdown.reward)
        resource_costs.append(breakdown.resource_cost)
        slo_violations.append(float(breakdown.slo_violation))

    return AutoscalingMetrics(
        policy_name=policy_name,
        total_reward=float(np.sum(rewards)),
        total_resource_cost=float(np.sum(resource_costs)),
        total_scaling_cost=0.0,
        slo_violation_rate=float(np.mean(slo_violations)),
        scale_action_count=0,
        mean_task_count=float(task_count),
        mean_overprovision_ratio=mean_overprovision_ratio(demand_array, capacity),
        final_task_count=task_count,
    )
