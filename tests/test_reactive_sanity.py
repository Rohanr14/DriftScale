"""Sanity tests for the reactive baseline used as the zero-forgetting reference."""

from __future__ import annotations

import numpy as np

from driftscale.baselines.reactive import ReactiveAutoscaler
from driftscale.envs.reward import RewardConfig


def test_reactive_evaluation_is_deterministic_for_identical_demand() -> None:
    """Reactive autoscaler is stateless across runs; two evaluations on the same demand
    must yield identical rewards. This is what guarantees reactive-baseline BWT == 0
    in the sensitivity summary."""
    rng = np.random.default_rng(0)
    demand = rng.uniform(0.5, 4.0, size=288).astype(np.float32)
    autoscaler = ReactiveAutoscaler(
        min_tasks=1,
        max_tasks=20,
        initial_tasks=1,
        capacity_per_task=1.0,
        reward_config=RewardConfig(),
    )
    first = autoscaler.evaluate(demand)
    second = autoscaler.evaluate(demand)
    assert first.total_reward == second.total_reward
    assert first.slo_violation_rate == second.slo_violation_rate


def test_reactive_bwt_across_stages_is_zero() -> None:
    """Simulate the reactive sanity row: evaluate on the 'initial' task before and
    'after' arbitrary stages of training (training does nothing for reactive). Final
    BWT must be exactly zero — the eval pipeline's signal that any non-zero BWT in
    the learned policies is real, not artifact."""
    rng = np.random.default_rng(1)
    task_1 = rng.uniform(0.5, 4.0, size=288).astype(np.float32)
    autoscaler = ReactiveAutoscaler(
        min_tasks=1,
        max_tasks=20,
        initial_tasks=1,
        capacity_per_task=1.0,
        reward_config=RewardConfig(),
    )
    reward_pre = autoscaler.evaluate(task_1).total_reward
    # Pretend several drift "stages" happened.
    for _ in range(5):
        other_task = rng.uniform(0.5, 4.0, size=288).astype(np.float32)
        autoscaler.evaluate(other_task)
    reward_post = autoscaler.evaluate(task_1).total_reward
    assert reward_post - reward_pre == 0.0
