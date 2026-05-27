"""Calibrate Azure-trace demand and baseline capacity before PPO training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from driftscale.baselines.reactive import ReactiveAutoscaler
from driftscale.baselines.static import evaluate_static_policy, static_task_count
from driftscale.envs.reward import RewardConfig
from driftscale.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default="configs/train/ppo.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config_path = Path(args.train_config)
    config = load_yaml(train_config_path)
    calibration_cfg = config.get("calibration", {})
    source_path = Path(
        calibration_cfg.get("source_trace_path", "results/caches/azure_v1_linear.csv")
    )
    if not source_path.exists():
        raise FileNotFoundError(f"{source_path} does not exist; run make preprocess first")

    source_trace = pd.read_csv(source_path)
    base_demand = source_trace["demand"].to_numpy(dtype=np.float32)
    env_cfg = config.get("env", {})
    reward_cfg = RewardConfig(**config.get("reward", {}))

    result = calibrate_static_p95(
        base_demand,
        min_tasks=int(env_cfg.get("min_tasks", 1)),
        max_tasks=int(env_cfg.get("max_tasks", 20)),
        target_min=float(calibration_cfg.get("target_slo_min", 0.01)),
        target_max=float(calibration_cfg.get("target_slo_max", 0.05)),
    )
    demand = base_demand * result["scale_factor"]
    capacity_per_task = float(result["capacity_per_task"])

    static_median_tasks = static_task_count(
        demand,
        quantile=0.50,
        capacity_per_task=capacity_per_task,
        min_tasks=int(env_cfg.get("min_tasks", 1)),
        max_tasks=int(env_cfg.get("max_tasks", 20)),
    )
    static_p95_tasks = static_task_count(
        demand,
        quantile=0.95,
        capacity_per_task=capacity_per_task,
        min_tasks=int(env_cfg.get("min_tasks", 1)),
        max_tasks=int(env_cfg.get("max_tasks", 20)),
    )
    baselines = [
        evaluate_static_policy(
            demand,
            task_count=static_median_tasks,
            policy_name="static_median",
            capacity_per_task=capacity_per_task,
            reward_config=reward_cfg,
        ).as_dict(),
        evaluate_static_policy(
            demand,
            task_count=static_p95_tasks,
            policy_name="static_p95",
            capacity_per_task=capacity_per_task,
            reward_config=reward_cfg,
        ).as_dict(),
        ReactiveAutoscaler(
            min_tasks=int(env_cfg.get("min_tasks", 1)),
            max_tasks=int(env_cfg.get("max_tasks", 20)),
            initial_tasks=int(env_cfg.get("initial_tasks", 4)),
            capacity_per_task=capacity_per_task,
            reward_config=reward_cfg,
        ).evaluate(demand).as_dict(),
    ]
    baseline_metrics = pd.DataFrame(baselines)
    baseline_metrics["total_cost"] = (
        baseline_metrics["total_resource_cost"] + baseline_metrics["total_scaling_cost"]
    )

    calibrated_trace = source_trace.copy()
    calibrated_trace["demand"] = demand.astype(np.float32)
    calibrated_trace["scale_factor"] = float(result["scale_factor"])
    calibrated_trace["capacity_per_task"] = capacity_per_task
    calibrated_trace_path = Path(
        calibration_cfg.get("calibrated_trace_path", "results/caches/azure_v1_calibrated.csv")
    )
    calibrated_trace_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated_trace.to_csv(calibrated_trace_path, index=False)

    metrics_path = Path(
        calibration_cfg.get("metrics_path", "results/calibration/baseline_metrics.csv")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_metrics.to_csv(metrics_path, index=False)
    summary_path = metrics_path.with_suffix(".json")
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    update_train_config(
        train_config_path,
        scale_factor=float(result["scale_factor"]),
        capacity_per_task=capacity_per_task,
        calibrated_trace_path=str(calibrated_trace_path),
        metrics_path=str(metrics_path),
    )

    static_p95_slo = baseline_metrics.loc[
        baseline_metrics["policy"] == "static_p95", "slo_violation_rate"
    ].iloc[0]
    print(f"Calibrated scale_factor: {result['scale_factor']:.6f}")
    print(f"Calibrated capacity_per_task: {capacity_per_task:.6f}")
    print(f"Static p95 SLO violation rate: {static_p95_slo:.3f}")
    print(baseline_metrics.to_string(index=False))
    print(f"Saved calibrated trace to {calibrated_trace_path}")
    print(f"Saved baseline metrics to {metrics_path}")
    print(f"Updated {train_config_path}")


def calibrate_static_p95(
    base_demand: np.ndarray,
    *,
    min_tasks: int,
    max_tasks: int,
    target_min: float,
    target_max: float,
) -> dict[str, float | int | bool]:
    """Search scale/capacity pairs until static p95 lands in the target SLO band."""
    if base_demand.ndim != 1 or base_demand.size < 2:
        raise ValueError("base_demand must be a one-dimensional array with at least two values")

    candidate_ratios = np.geomspace(0.001, 24.0, 1600)
    capacity_candidates = np.linspace(0.5, 2.0, 31)
    target_mid = (target_min + target_max) / 2.0
    best: dict[str, float | int | bool] | None = None
    best_score = float("inf")

    for capacity_per_task in capacity_candidates:
        for ratio in candidate_ratios:
            scale_factor = float(ratio * capacity_per_task)
            demand = base_demand * scale_factor
            task_count = static_task_count(
                demand,
                quantile=0.95,
                capacity_per_task=float(capacity_per_task),
                min_tasks=min_tasks,
                max_tasks=max_tasks,
            )
            capacity = task_count * capacity_per_task
            slo = float(np.mean(demand > capacity))
            in_target = target_min <= slo <= target_max
            task_margin = 0.0 if min_tasks < task_count < max_tasks else 0.25
            capacity_regularizer = 0.001 * abs(float(capacity_per_task) - 1.0)
            score = abs(slo - target_mid) + task_margin + capacity_regularizer
            if in_target:
                score -= 1.0

            if score < best_score:
                best_score = score
                best = {
                    "scale_factor": scale_factor,
                    "capacity_per_task": float(capacity_per_task),
                    "static_p95_task_count": int(task_count),
                    "static_p95_slo_violation_rate": slo,
                    "target_min": target_min,
                    "target_max": target_max,
                    "in_target": in_target,
                }

    if best is None:
        raise RuntimeError("calibration search produced no candidates")
    if not bool(best["in_target"]):
        raise RuntimeError(
            "could not calibrate static p95 into target SLO band; "
            f"best SLO={best['static_p95_slo_violation_rate']:.3f}"
        )
    return best


def update_train_config(
    train_config_path: Path,
    *,
    scale_factor: float,
    capacity_per_task: float,
    calibrated_trace_path: str,
    metrics_path: str,
) -> None:
    config = load_yaml(train_config_path)
    config.setdefault("trace", {})["path"] = calibrated_trace_path
    config["trace"]["demand_column"] = config["trace"].get("demand_column", "demand")
    config.setdefault("env", {})["scale_factor"] = scale_factor
    config["env"]["capacity_per_task"] = capacity_per_task
    config.setdefault("calibration", {})["scale_factor"] = scale_factor
    config["calibration"]["capacity_per_task"] = capacity_per_task
    config["calibration"]["calibrated_trace_path"] = calibrated_trace_path
    config["calibration"]["metrics_path"] = metrics_path

    with train_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


if __name__ == "__main__":
    main()
