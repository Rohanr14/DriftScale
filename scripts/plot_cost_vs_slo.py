"""Cost vs. SLO Pareto plot covering Static, Reactive, and PPO variants.

Evaluates every policy on the SAME held-out trace (the final Azure checkpoint) so the
plot compares like-for-like behavior under drift. PPO variants are averaged across the
sensitivity-suite seeds; error bars on both axes show the seed-level spread.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from driftscale.agents.train_ppo import build_vecnormalize_env
from driftscale.baselines.reactive import ReactiveAutoscaler
from driftscale.baselines.static import evaluate_static_policy, static_task_count
from driftscale.envs.reward import RewardConfig
from driftscale.utils.config import load_yaml

PPO_METHODS = ("naive", "replay")
METHOD_LABELS = {
    "static_median": "Static median",
    "static_p95": "Static p95",
    "reactive_threshold": "Reactive (threshold)",
    "ppo_naive": "PPO + Naive Fine-Tune",
    "ppo_replay": "PPO + Replay",
}
METHOD_MARKERS = {
    "static_median": "s",
    "static_p95": "s",
    "reactive_threshold": "D",
    "ppo_naive": "o",
    "ppo_replay": "o",
}
METHOD_COLORS = {
    "static_median": "tab:gray",
    "static_p95": "tab:olive",
    "reactive_threshold": "tab:green",
    "ppo_naive": "tab:red",
    "ppo_replay": "tab:blue",
}


@dataclass(frozen=True)
class PolicyPoint:
    """One scatter point on the cost-vs-SLO plot."""

    name: str
    slo_violation_rate: float
    slo_violation_rate_err: float
    total_cost: float
    total_cost_err: float
    n_seeds: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default="linear")
    parser.add_argument("--final-stage", type=int, default=125)
    parser.add_argument("--sensitivity-dir", default="results/sensitivity")
    parser.add_argument("--ppo-config", default="configs/train/ppo.yaml")
    parser.add_argument("--env-config")
    parser.add_argument("--output-path", default="media/cost_vs_slo.png")
    parser.add_argument("--seeds", nargs="+", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensitivity_dir = Path(args.sensitivity_dir)
    mapping_dir = sensitivity_dir / args.mapping
    final_trace_path = mapping_dir / f"task_{args.final_stage}_calibrated.csv"
    if not final_trace_path.exists():
        raise FileNotFoundError(
            f"{final_trace_path} missing; run `make sensitivity-suite` first."
        )

    env_config_path = Path(args.env_config or f"configs/env/{args.mapping}.yaml")
    base_config = build_base_config(
        ppo_config_path=Path(args.ppo_config),
        env_config_path=env_config_path,
        calibrated_trace_path=final_trace_path,
    )
    seeds = args.seeds or detect_seeds(mapping_dir)
    if not seeds:
        raise RuntimeError(
            f"No seed directories found under {mapping_dir}; "
            "pass --seeds or rerun sensitivity-suite."
        )

    demand_trace = pd.read_csv(final_trace_path)
    demand = demand_trace["demand"].to_numpy(dtype=np.float32)
    capacity_per_task = float(demand_trace["capacity_per_task"].iloc[0])

    points = [
        evaluate_static_point(
            demand,
            policy_name="static_median",
            quantile=0.50,
            config=base_config,
            capacity_per_task=capacity_per_task,
        ),
        evaluate_static_point(
            demand,
            policy_name="static_p95",
            quantile=0.95,
            config=base_config,
            capacity_per_task=capacity_per_task,
        ),
        evaluate_reactive_point(
            demand,
            config=base_config,
            capacity_per_task=capacity_per_task,
        ),
    ]
    for method in PPO_METHODS:
        ppo_point = evaluate_ppo_method_point(
            method=method,
            mapping_dir=mapping_dir,
            stage=args.final_stage,
            seeds=seeds,
            base_config=base_config,
            demand=demand,
        )
        if ppo_point is not None:
            points.append(ppo_point)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_cost_vs_slo_plot(points, output_path=output_path)
    print(format_points_table(points))
    print(f"Saved {output_path}")


def detect_seeds(mapping_dir: Path) -> list[int]:
    seeds = []
    for child in sorted(mapping_dir.glob("seed_*")):
        try:
            seeds.append(int(child.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return seeds


def build_base_config(
    *,
    ppo_config_path: Path,
    env_config_path: Path,
    calibrated_trace_path: Path,
) -> dict[str, Any]:
    config = load_yaml(ppo_config_path)
    env_config = load_yaml(env_config_path)
    if "reward" in env_config:
        config.setdefault("reward", {}).update(env_config["reward"])
    if "env" in env_config:
        config.setdefault("env", {}).update(env_config["env"])
    trace = pd.read_csv(calibrated_trace_path)
    capacity_per_task = float(trace["capacity_per_task"].iloc[0])
    config.setdefault("env", {})["capacity_per_task"] = capacity_per_task
    config["env"].setdefault("initial_tasks", 1)
    config.setdefault("eval", {})["deterministic"] = True
    return config


def evaluate_static_point(
    demand: np.ndarray,
    *,
    policy_name: str,
    quantile: float,
    config: dict[str, Any],
    capacity_per_task: float,
) -> PolicyPoint:
    env_cfg = config.get("env", {})
    reward_cfg = RewardConfig(**config.get("reward", {}))
    task_count = static_task_count(
        demand,
        quantile=quantile,
        capacity_per_task=capacity_per_task,
        min_tasks=int(env_cfg.get("min_tasks", 1)),
        max_tasks=int(env_cfg.get("max_tasks", 20)),
    )
    metrics = evaluate_static_policy(
        demand,
        task_count=task_count,
        policy_name=policy_name,
        capacity_per_task=capacity_per_task,
        reward_config=reward_cfg,
    )
    total_cost = metrics.total_resource_cost + metrics.total_scaling_cost
    return PolicyPoint(
        name=policy_name,
        slo_violation_rate=float(metrics.slo_violation_rate),
        slo_violation_rate_err=0.0,
        total_cost=float(total_cost),
        total_cost_err=0.0,
        n_seeds=1,
    )


def evaluate_reactive_point(
    demand: np.ndarray,
    *,
    config: dict[str, Any],
    capacity_per_task: float,
) -> PolicyPoint:
    env_cfg = config.get("env", {})
    reward_cfg = RewardConfig(**config.get("reward", {}))
    autoscaler = ReactiveAutoscaler(
        min_tasks=int(env_cfg.get("min_tasks", 1)),
        max_tasks=int(env_cfg.get("max_tasks", 20)),
        initial_tasks=int(env_cfg.get("initial_tasks", 1)),
        capacity_per_task=capacity_per_task,
        reward_config=reward_cfg,
    )
    metrics = autoscaler.evaluate(demand)
    total_cost = metrics.total_resource_cost + metrics.total_scaling_cost
    return PolicyPoint(
        name="reactive_threshold",
        slo_violation_rate=float(metrics.slo_violation_rate),
        slo_violation_rate_err=0.0,
        total_cost=float(total_cost),
        total_cost_err=0.0,
        n_seeds=1,
    )


def evaluate_ppo_method_point(
    *,
    method: str,
    mapping_dir: Path,
    stage: int,
    seeds: list[int],
    base_config: dict[str, Any],
    demand: np.ndarray,
) -> PolicyPoint | None:
    method_dir_template = (
        f"naive_stage_{stage}" if method == "naive" else f"replay_stage_{stage}"
    )
    slo_rates: list[float] = []
    costs: list[float] = []
    for seed in seeds:
        method_dir = mapping_dir / f"seed_{seed}" / method_dir_template
        model_path = method_dir / "model.zip"
        vecnormalize_path = method_dir / "vecnormalize.pkl"
        if not model_path.exists() or not vecnormalize_path.exists():
            continue
        env = build_vecnormalize_env(
            base_config,
            demand=demand,
            seed=int(base_config.get("seed", 0)),
            vecnormalize_path=vecnormalize_path,
            training=False,
        )
        env.norm_reward = False
        model = MaskablePPO.load(str(model_path), env=env)
        metrics = evaluate_episode(model, env)
        slo_rates.append(metrics["slo_violation_rate"])
        costs.append(metrics["total_cost"])
    if not slo_rates:
        print(f"  {method}: no model artifacts found; skipping point.")
        return None
    return PolicyPoint(
        name=f"ppo_{method}",
        slo_violation_rate=float(np.mean(slo_rates)),
        slo_violation_rate_err=float(np.std(slo_rates, ddof=1)) if len(slo_rates) > 1 else 0.0,
        total_cost=float(np.mean(costs)),
        total_cost_err=float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
        n_seeds=len(slo_rates),
    )


def evaluate_episode(model: MaskablePPO, env) -> dict[str, float]:
    obs = env.reset()
    done = np.array([False])
    resource_cost = 0.0
    scaling_cost = 0.0
    slo_violations = 0
    steps = 0
    while not bool(done[0]):
        masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        obs, _rewards, done, infos = env.step(action)
        info = infos[0]
        resource_cost += float(info["resource_cost"])
        scaling_cost += float(info["scaling_action_penalty"])
        slo_violations += int(info["slo_violation"])
        steps += 1
    return {
        "slo_violation_rate": slo_violations / max(steps, 1),
        "total_cost": resource_cost + scaling_cost,
    }


def write_cost_vs_slo_plot(points: list[PolicyPoint], *, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.6, 5.4))
    for point in points:
        slo_pct = point.slo_violation_rate * 100.0
        slo_err = point.slo_violation_rate_err * 100.0
        axis.errorbar(
            slo_pct,
            point.total_cost,
            xerr=slo_err if slo_err > 0 else None,
            yerr=point.total_cost_err if point.total_cost_err > 0 else None,
            marker=METHOD_MARKERS.get(point.name, "o"),
            markersize=10,
            color=METHOD_COLORS.get(point.name, "tab:purple"),
            ecolor="0.55",
            capsize=3.0,
            linewidth=0,
            elinewidth=1.0,
            label=METHOD_LABELS.get(point.name, point.name),
        )
    apply_offsets_and_annotate(axis, points)
    frontier = compute_pareto_frontier(points)
    if len(frontier) >= 2:
        xs = [p.slo_violation_rate * 100.0 for p in frontier]
        ys = [p.total_cost for p in frontier]
        axis.plot(xs, ys, color="0.4", linestyle="--", linewidth=1.2, label="Pareto frontier")
    axis.set_xlabel("SLO violation rate (%)")
    axis.set_ylabel("Total simulated cost (lower is better)")
    axis.set_title("Cost vs. SLO Pareto Frontier — Final Azure Checkpoint")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def apply_offsets_and_annotate(axis, points: list[PolicyPoint]) -> None:
    """Hand-tuned text offsets so the static-* labels never collide."""
    offsets = {
        "static_median": (10, -16),
        "static_p95": (10, 18),
        "reactive_threshold": (12, -8),
        "ppo_naive": (-14, 18),
        "ppo_replay": (-14, -22),
    }
    for point in points:
        x = point.slo_violation_rate * 100.0
        y = point.total_cost
        offset = offsets.get(point.name, (10, 10))
        axis.annotate(
            METHOD_LABELS.get(point.name, point.name),
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "alpha": 0.25, "linewidth": 0.7},
        )


def compute_pareto_frontier(points: list[PolicyPoint]) -> list[PolicyPoint]:
    """Return non-dominated points (lower SLO, lower cost both better)."""
    sorted_points = sorted(points, key=lambda p: (p.slo_violation_rate, p.total_cost))
    frontier: list[PolicyPoint] = []
    min_cost = float("inf")
    for point in sorted_points:
        if point.total_cost < min_cost:
            frontier.append(point)
            min_cost = point.total_cost
    return frontier


def format_points_table(points: list[PolicyPoint]) -> str:
    rows = [
        "policy            | SLO%  |  cost  | n_seeds",
        "-" * 50,
    ]
    for point in points:
        rows.append(
            f"{point.name:<17} | {point.slo_violation_rate * 100:5.2f} | "
            f"{point.total_cost:7.2f} | {point.n_seeds:>2}"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
