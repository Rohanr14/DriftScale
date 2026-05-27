import numpy as np
import pytest

from driftscale.envs.cost_env import DriftScaleEnv
from driftscale.envs.spaces import ACTION_DELTAS
from driftscale.traces.synthetic import generate_synthetic_episode


def make_env(initial_tasks: int = 4) -> DriftScaleEnv:
    episode = generate_synthetic_episode(regime="bursty", length=64, seed=5)
    return DriftScaleEnv(episode.demand, initial_tasks=initial_tasks)


def test_reset_observation_matches_space() -> None:
    env = make_env()
    observation, info = env.reset(seed=1)

    assert observation.shape == env.observation_space.shape
    assert env.observation_space.contains(observation)
    assert info["task_count"] == 4


def test_action_masks_block_boundary_actions() -> None:
    min_env = make_env(initial_tasks=1)
    min_env.reset()
    max_env = make_env(initial_tasks=20)
    max_env.reset()

    assert not min_env.action_masks()[np.where(ACTION_DELTAS == -2)[0][0]]
    assert not min_env.action_masks()[np.where(ACTION_DELTAS == -1)[0][0]]
    assert not max_env.action_masks()[np.where(ACTION_DELTAS == 1)[0][0]]
    assert not max_env.action_masks()[np.where(ACTION_DELTAS == 2)[0][0]]


def test_invalid_masked_action_raises() -> None:
    env = make_env(initial_tasks=1)
    env.reset()
    invalid_action = int(np.where(ACTION_DELTAS == -1)[0][0])

    with pytest.raises(ValueError, match="invalid"):
        env.step(invalid_action)


def test_masked_random_policy_completes_episode() -> None:
    rng = np.random.default_rng(9)
    env = make_env()
    env.reset(seed=9)
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = int(rng.choice(np.flatnonzero(env.action_masks())))
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        assert info["task_count"] >= env.min_tasks
        assert info["task_count"] <= env.max_tasks

    assert env.index == len(env.demand)
    assert np.isfinite(total_reward)
