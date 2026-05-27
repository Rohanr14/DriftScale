"""Reactive threshold autoscaling baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from driftscale.baselines.metrics import AutoscalingMetrics, mean_overprovision_ratio
from driftscale.envs.reward import RewardConfig, calculate_reward


@dataclass(frozen=True)
class ReactiveAutoscaler:
    """CPU-threshold autoscaler matching the project bible's reference baseline."""

    min_tasks: int = 1
    max_tasks: int = 20
    initial_tasks: int = 4
    capacity_per_task: float = 1.0
    high_threshold: float = 0.70
    low_threshold: float = 0.30
    scale_up_after: int = 1
    scale_down_after: int = 6
    scale_step: int = 2
    reward_config: RewardConfig = RewardConfig()

    def evaluate(self, demand: np.ndarray | list[float]) -> AutoscalingMetrics:
        demand_array = np.asarray(demand, dtype=np.float32)
        task_count = int(np.clip(self.initial_tasks, self.min_tasks, self.max_tasks))
        high_count = 0
        low_count = 0

        task_history: list[int] = []
        capacity_history: list[float] = []
        rewards: list[float] = []
        resource_costs: list[float] = []
        scaling_costs: list[float] = []
        slo_violations: list[float] = []
        scale_action_count = 0

        for demand_t in demand_array:
            old_task_count = task_count
            observed_cpu = float(demand_t / max(task_count * self.capacity_per_task, 1e-8))

            if observed_cpu > self.high_threshold:
                high_count += 1
                low_count = 0
            elif observed_cpu < self.low_threshold:
                low_count += 1
                high_count = 0
            else:
                high_count = 0
                low_count = 0

            if high_count >= self.scale_up_after:
                task_count = min(self.max_tasks, task_count + self.scale_step)
                high_count = 0
            elif low_count >= self.scale_down_after:
                task_count = max(self.min_tasks, task_count - 1)
                low_count = 0

            if task_count != old_task_count:
                scale_action_count += 1

            capacity = task_count * self.capacity_per_task
            breakdown = calculate_reward(
                demand=float(demand_t),
                capacity=capacity,
                old_task_count=old_task_count,
                new_task_count=task_count,
                config=self.reward_config,
            )

            task_history.append(task_count)
            capacity_history.append(capacity)
            rewards.append(breakdown.reward)
            resource_costs.append(breakdown.resource_cost)
            scaling_costs.append(breakdown.scaling_action_penalty)
            slo_violations.append(float(breakdown.slo_violation))

        return AutoscalingMetrics(
            policy_name="reactive_threshold",
            total_reward=float(np.sum(rewards)),
            total_resource_cost=float(np.sum(resource_costs)),
            total_scaling_cost=float(np.sum(scaling_costs)),
            slo_violation_rate=float(np.mean(slo_violations)),
            scale_action_count=scale_action_count,
            mean_task_count=float(np.mean(task_history)),
            mean_overprovision_ratio=mean_overprovision_ratio(
                demand_array,
                np.asarray(capacity_history, dtype=np.float32),
            ),
            final_task_count=int(task_history[-1]),
        )

