"""Run a masked random policy through DriftScaleEnv."""

from __future__ import annotations

import argparse

import numpy as np

from driftscale.envs.cost_env import DriftScaleEnv
from driftscale.envs.reward import RewardConfig
from driftscale.traces.synthetic import generate_synthetic_episode
from driftscale.utils.config import load_yaml
from driftscale.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env/synthetic.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    episode = generate_synthetic_episode(
        regime=config.get("regime", "bursty"),
        length=int(config.get("length", 288)),
        seed=seed,
        noise=float(config.get("noise", 0.04)),
        step_minutes=int(config.get("step_minutes", 5)),
    )
    env_cfg = config.get("env", {})
    env = DriftScaleEnv(
        episode.demand,
        min_tasks=int(env_cfg.get("min_tasks", 1)),
        max_tasks=int(env_cfg.get("max_tasks", 20)),
        initial_tasks=int(env_cfg.get("initial_tasks", 4)),
        capacity_per_task=float(env_cfg.get("capacity_per_task", 1.0)),
        strict_action_mask=bool(env_cfg.get("strict_action_mask", True)),
        step_minutes=int(config.get("step_minutes", 5)),
        reward_config=RewardConfig(**config.get("reward", {})),
    )

    _, _ = env.reset(seed=seed)
    total_reward = 0.0
    slo_violations = []
    scale_actions = 0
    steps = 0

    terminated = False
    truncated = False
    while not (terminated or truncated):
        valid_actions = np.flatnonzero(env.action_masks())
        action = int(rng.choice(valid_actions))
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        slo_violations.append(float(info["slo_violation"]))
        scale_actions += int(info["action_delta"] != 0)
        steps += 1

    print(f"Random masked policy completed {steps} steps")
    print(f"Total reward: {total_reward:.3f}")
    print(f"SLO violation rate: {np.mean(slo_violations):.3f}")
    print(f"Scale action count: {scale_actions}")


if __name__ == "__main__":
    main()

