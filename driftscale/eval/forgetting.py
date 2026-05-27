"""Forgetting and backward-transfer metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForgettingMetrics:
    """Task-A retention after training on Task B."""

    method: str
    task_a_reward_before_b: float
    task_a_reward_after_b: float

    @property
    def backward_transfer(self) -> float:
        """BWT: old-task reward after new-task training minus old-task reward before it."""
        return self.task_a_reward_after_b - self.task_a_reward_before_b

    @property
    def forgetting(self) -> float:
        """Positive values mean Task-A reward decreased after Task-B training."""
        return max(0.0, -self.backward_transfer)

    @property
    def retention_ratio(self) -> float:
        denominator = abs(self.task_a_reward_before_b)
        if denominator == 0.0:
            return 1.0
        return max(0.0, 1.0 - (self.forgetting / denominator))

    def as_dict(self) -> dict[str, float | str]:
        return {
            "method": self.method,
            "task_a_reward_before_b": self.task_a_reward_before_b,
            "task_a_reward_after_b": self.task_a_reward_after_b,
            "bwt": self.backward_transfer,
            "forgetting": self.forgetting,
            "retention_ratio": self.retention_ratio,
        }


def calculate_bwt(*, reward_before: float, reward_after: float) -> float:
    """Return backward transfer from two old-task reward measurements."""
    return reward_after - reward_before


def mean_prior_bwt(
    reward_matrix: np.ndarray | list[list[float]],
    *,
    stage_index: int | None = None,
) -> float:
    """Continual-learning BWT averaged over all prior tasks at the chosen stage.

    ``reward_matrix[i][k]`` is the reward of the stage-``k`` policy evaluated on task ``i``,
    using zero-indexed stages and tasks. The BWT at stage ``k`` is the mean over
    ``i < k`` of ``R_{i, k} - R_{i, i}``, capturing average regression on every previously
    learned task.

    ``stage_index`` defaults to the final stage. Returns ``nan`` when there are no prior
    tasks (e.g. the first stage).
    """
    matrix = np.asarray(reward_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("reward_matrix must be 2-dimensional [task, stage]")
    n_tasks, n_stages = matrix.shape
    if n_tasks != n_stages:
        raise ValueError("reward_matrix must be square (one row per task per stage)")
    stage = n_stages - 1 if stage_index is None else int(stage_index)
    if stage < 0 or stage >= n_stages:
        raise IndexError(f"stage_index {stage} out of bounds for {n_stages} stages")
    if stage == 0:
        return float("nan")
    differences = [matrix[i, stage] - matrix[i, i] for i in range(stage)]
    return float(np.mean(differences))


def rolling_mean_prior_bwt(reward_matrix: np.ndarray | list[list[float]]) -> np.ndarray:
    """Return ``mean_prior_bwt`` for every stage in a square reward matrix."""
    matrix = np.asarray(reward_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("reward_matrix must be 2-dimensional [task, stage]")
    n_stages = matrix.shape[1]
    return np.asarray(
        [mean_prior_bwt(matrix, stage_index=stage) for stage in range(n_stages)],
        dtype=np.float64,
    )
