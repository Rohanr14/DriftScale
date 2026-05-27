import math

import numpy as np
import pytest

from driftscale.eval.forgetting import (
    ForgettingMetrics,
    calculate_bwt,
    mean_prior_bwt,
    rolling_mean_prior_bwt,
)


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


def test_mean_prior_bwt_at_first_stage_is_nan() -> None:
    matrix = np.array([[1.0, 0.5, 0.4], [0.0, 1.0, 0.6], [0.0, 0.0, 1.0]])
    assert math.isnan(mean_prior_bwt(matrix, stage_index=0))


def test_mean_prior_bwt_averages_drops_on_all_prior_tasks() -> None:
    # Square [task, stage] matrix; diagonal is the "just-trained" reward per task.
    # Stage 2 means mean of (R_{0, 2} - R_{0, 0}) and (R_{1, 2} - R_{1, 1}).
    matrix = np.array(
        [
            [10.0, 8.0, 4.0],
            [0.0, 9.0, 6.0],
            [0.0, 0.0, 11.0],
        ]
    )
    # ((4 - 10) + (6 - 9)) / 2 = (-6 + -3) / 2 = -4.5
    assert mean_prior_bwt(matrix, stage_index=2) == pytest.approx(-4.5)


def test_mean_prior_bwt_defaults_to_final_stage() -> None:
    matrix = np.array(
        [
            [10.0, 8.0, 4.0],
            [0.0, 9.0, 6.0],
            [0.0, 0.0, 11.0],
        ]
    )
    assert mean_prior_bwt(matrix) == pytest.approx(-4.5)


def test_rolling_mean_prior_bwt_returns_one_value_per_stage() -> None:
    matrix = np.array(
        [
            [5.0, 3.0, 1.0],
            [0.0, 6.0, 2.0],
            [0.0, 0.0, 7.0],
        ]
    )
    rolling = rolling_mean_prior_bwt(matrix)
    assert math.isnan(rolling[0])
    assert rolling[1] == pytest.approx(-2.0)  # (3 - 5) / 1
    assert rolling[2] == pytest.approx(-4.0)  # ((1 - 5) + (2 - 6)) / 2


def test_mean_prior_bwt_rejects_non_square_matrices() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(ValueError):
        mean_prior_bwt(matrix)
