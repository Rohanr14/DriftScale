"""Evaluate Phase 1 baselines on the synthetic workload."""

from __future__ import annotations

import argparse
from pathlib import Path

from driftscale.envs.reward import RewardConfig
from driftscale.eval.report import phase1_baseline_report
from driftscale.traces.synthetic import generate_synthetic_episode
from driftscale.utils.config import load_yaml
from driftscale.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env/synthetic.yaml")
    parser.add_argument("--output-dir", default="results/phase1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config.get("seed", 0))
    seed_everything(seed)

    episode = generate_synthetic_episode(
        regime=config.get("regime", "bursty"),
        length=int(config.get("length", 288)),
        seed=seed,
        noise=float(config.get("noise", 0.04)),
        step_minutes=int(config.get("step_minutes", 5)),
    )

    env_cfg = config.get("env", {})
    reward_cfg = RewardConfig(**config.get("reward", {}))
    report = phase1_baseline_report(
        episode.demand,
        min_tasks=int(env_cfg.get("min_tasks", 1)),
        max_tasks=int(env_cfg.get("max_tasks", 20)),
        initial_tasks=int(env_cfg.get("initial_tasks", 4)),
        capacity_per_task=float(env_cfg.get("capacity_per_task", 1.0)),
        reward_config=reward_cfg,
        reactive_kwargs=config.get("reactive", {}),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "synthetic_baselines.csv"
    report.to_csv(csv_path, index=False)

    print(f"Synthetic regime: {episode.regime}")
    print(f"Steps: {len(episode.demand)}")
    print(report.to_string(index=False))
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()

