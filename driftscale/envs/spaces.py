"""Shared action and observation space helpers."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

ACTION_DELTAS = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
OBSERVATION_SIZE = 12


def build_observation_space() -> spaces.Box:
    """Return the normalized observation space used by DriftScaleEnv."""
    low = np.array(
        [
            0.0,  # current_cpu_util
            0.0,  # avg_cpu_5
            0.0,  # avg_cpu_15
            0.0,  # avg_cpu_60
            0.0,  # max_cpu_15
            0.0,  # std_cpu_15
            0.0,  # current_task_count
            -1.0,  # previous_action
            -1.0,  # time_of_day_sin
            -1.0,  # time_of_day_cos
            0.0,  # recent_slo_violation_rate
            0.0,  # recent_scale_action_count
        ],
        dtype=np.float32,
    )
    high = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )
    return spaces.Box(low=low, high=high, shape=(OBSERVATION_SIZE,), dtype=np.float32)


def action_mask(task_count: int, min_tasks: int, max_tasks: int) -> np.ndarray:
    """Mask actions that would leave the allowed task-count range."""
    next_counts = task_count + ACTION_DELTAS
    return ((next_counts >= min_tasks) & (next_counts <= max_tasks)).astype(bool)


def normalize_action_delta(delta: int) -> float:
    """Map a scale delta into [-1, 1]."""
    return float(delta / np.max(np.abs(ACTION_DELTAS)))


def normalize_task_count(task_count: int, min_tasks: int, max_tasks: int) -> float:
    """Map the current task count into [0, 1]."""
    if max_tasks == min_tasks:
        return 0.0
    return float((task_count - min_tasks) / (max_tasks - min_tasks))

