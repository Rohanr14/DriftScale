from driftscale.eval.forgetting import ForgettingMetrics, calculate_bwt


def test_bwt_and_forgetting_signs() -> None:
    metrics = ForgettingMetrics(
        method="naive",
        task_a_reward_before_b=-10.0,
        task_a_reward_after_b=-15.0,
    )

    assert calculate_bwt(reward_before=-10.0, reward_after=-15.0) == -5.0
    assert metrics.backward_transfer == -5.0
    assert metrics.forgetting == 5.0
    assert metrics.retention_ratio == 0.5


def test_positive_bwt_has_zero_forgetting() -> None:
    metrics = ForgettingMetrics(
        method="replay",
        task_a_reward_before_b=-10.0,
        task_a_reward_after_b=-8.0,
    )

    assert metrics.backward_transfer == 2.0
    assert metrics.forgetting == 0.0
    assert metrics.retention_ratio == 1.0
