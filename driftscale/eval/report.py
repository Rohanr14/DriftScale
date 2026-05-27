"""Phase 1 evaluation reports."""

from __future__ import annotations

import pandas as pd

from driftscale.baselines.reactive import ReactiveAutoscaler
from driftscale.baselines.static import evaluate_static_policy, static_task_count
from driftscale.envs.reward import RewardConfig


def phase1_baseline_report(
    demand,
    *,
    min_tasks: int = 1,
    max_tasks: int = 20,
    initial_tasks: int = 4,
    capacity_per_task: float = 1.0,
    reward_config: RewardConfig | None = None,
    reactive_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Evaluate static median, static p95, and reactive baselines."""
    reward_cfg = reward_config or RewardConfig()
    median_tasks = static_task_count(
        demand,
        quantile=0.50,
        capacity_per_task=capacity_per_task,
        min_tasks=min_tasks,
        max_tasks=max_tasks,
    )
    p95_tasks = static_task_count(
        demand,
        quantile=0.95,
        capacity_per_task=capacity_per_task,
        min_tasks=min_tasks,
        max_tasks=max_tasks,
    )

    static_median = evaluate_static_policy(
        demand,
        task_count=median_tasks,
        policy_name="static_median",
        capacity_per_task=capacity_per_task,
        reward_config=reward_cfg,
    )
    static_p95 = evaluate_static_policy(
        demand,
        task_count=p95_tasks,
        policy_name="static_p95",
        capacity_per_task=capacity_per_task,
        reward_config=reward_cfg,
    )
    reactive = ReactiveAutoscaler(
        min_tasks=min_tasks,
        max_tasks=max_tasks,
        initial_tasks=initial_tasks,
        capacity_per_task=capacity_per_task,
        reward_config=reward_cfg,
        **(reactive_kwargs or {}),
    ).evaluate(demand)

    rows = [static_median.as_dict(), static_p95.as_dict(), reactive.as_dict()]
    return pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)

