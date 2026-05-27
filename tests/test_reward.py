from driftscale.envs.reward import RewardConfig, calculate_reward


def test_slo_penalty_makes_underprovisioned_reward_worse() -> None:
    cfg = RewardConfig()
    enough_capacity = calculate_reward(
        demand=5.0,
        capacity=6.0,
        old_task_count=6,
        new_task_count=6,
        config=cfg,
    )
    underprovisioned = calculate_reward(
        demand=5.0,
        capacity=2.0,
        old_task_count=2,
        new_task_count=2,
        config=cfg,
    )

    assert underprovisioned.slo_violation
    assert underprovisioned.reward < enough_capacity.reward


def test_scaling_penalty_is_zero_when_task_count_is_unchanged() -> None:
    unchanged = calculate_reward(demand=3.0, capacity=3.0, old_task_count=3, new_task_count=3)
    changed = calculate_reward(demand=3.0, capacity=4.0, old_task_count=3, new_task_count=4)

    assert unchanged.scaling_action_penalty == 0.0
    assert changed.scaling_action_penalty > 0.0

