"""Plot Naive vs Replay rollouts side-by-side on the same Task-B demand trace.

A single seed × stage produces two policies (the naive_stage_{ckpt} and
replay_stage_{ckpt} model artifacts written by ``run_sensitivity_analysis.py``). Both are
rolled out deterministically on the same calibrated trace; the figure shows the agent
capacity, the reactive reference, the demand, the SLO violations, and the totals per
panel.

If a method's artifacts are missing the corresponding panel is skipped with a warning,
so this script can also produce a one-policy figure when only one method has been run.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from plot_episode_rollout import (
    add_rollout_diagnostics,
    collect_rollout,
    infer_capacity_per_task,
    load_rollout_input,
    rollout_config,
)
from sb3_contrib import MaskablePPO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from driftscale.agents.train_ppo import build_vecnormalize_env

METHOD_DIR_TEMPLATE = {
    "naive": "naive_stage_{stage}",
    "replay": "replay_stage_{stage}",
}
METHOD_TITLE = {
    "naive": "Naive Fine-Tuning",
    "replay": "PPO + Replay",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default="linear")
    parser.add_argument("--stage", type=int, default=125)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[1, 25, 50, 75, 100, 125])
    parser.add_argument(
        "--rollout-mode",
        choices=("curriculum", "stage"),
        default="stage",
        help=(
            "stage = roll out only the final checkpoint trace (clearer comparison). "
            "curriculum = roll out the whole six-checkpoint sequence."
        ),
    )
    parser.add_argument("--sensitivity-dir", default="results/sensitivity")
    parser.add_argument("--train-config", default="configs/train/ppo_replay.yaml")
    parser.add_argument("--env-config")
    parser.add_argument(
        "--methods", nargs="+", default=["naive", "replay"], choices=["naive", "replay"]
    )
    parser.add_argument("--output-path", default="media/episode_rollout_comparison.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensitivity_dir = Path(args.sensitivity_dir)
    mapping_dir = sensitivity_dir / args.mapping
    env_config_path = Path(args.env_config or f"configs/env/{args.mapping}.yaml")
    train_config_path = Path(args.train_config)

    if args.rollout_mode == "stage":
        trace_paths = [mapping_dir / f"task_{args.stage}_calibrated.csv"]
    else:
        trace_paths = [
            mapping_dir / f"task_{checkpoint}_calibrated.csv" for checkpoint in args.checkpoints
        ]
    for trace_path in trace_paths:
        if not trace_path.exists():
            raise FileNotFoundError(
                f"{trace_path} does not exist; run make sensitivity-suite first"
            )

    base_config = rollout_config(
        train_config_path=train_config_path,
        env_config_path=env_config_path,
        capacity_trace_path=trace_paths[0],
    )

    rollouts: dict[str, pd.DataFrame] = {}
    for method in args.methods:
        method_dir = mapping_dir / f"seed_{args.seed}" / METHOD_DIR_TEMPLATE[method].format(
            stage=args.stage
        )
        model_path = method_dir / "model.zip"
        vecnormalize_path = method_dir / "vecnormalize.pkl"
        if not model_path.exists() or not vecnormalize_path.exists():
            print(
                f"Skipping {method}: missing artifacts at {method_dir}; run sensitivity-suite "
                "with these methods first."
            )
            continue
        rollout = collect_method_rollout(
            method=method,
            train_config_path=train_config_path,
            env_config_path=env_config_path,
            trace_paths=trace_paths,
            checkpoints=args.checkpoints if args.rollout_mode == "curriculum" else [args.stage],
            model_path=model_path,
            vecnormalize_path=vecnormalize_path,
            base_config=base_config,
        )
        rollouts[method] = rollout

    if not rollouts:
        raise RuntimeError("No methods produced rollouts; nothing to plot.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_comparison_plot(
        rollouts,
        output_path=output_path,
        config=base_config,
        title_suffix=(
            f"stage {args.stage} ({args.mapping}, seed {args.seed})"
            if args.rollout_mode == "stage"
            else f"curriculum ({args.mapping}, seed {args.seed})"
        ),
    )
    print(f"Saved comparison plot to {output_path}")


def collect_method_rollout(
    *,
    method: str,
    train_config_path: Path,
    env_config_path: Path,
    trace_paths: list[Path],
    checkpoints: list[int],
    model_path: Path,
    vecnormalize_path: Path,
    base_config: dict[str, Any],
) -> pd.DataFrame:
    config = copy.deepcopy(base_config)
    rollout_input = load_rollout_input(trace_paths=trace_paths, checkpoints=checkpoints)
    demand = rollout_input["demand"].to_numpy(dtype=np.float32)
    env = build_vecnormalize_env(
        config,
        demand=demand,
        seed=int(config.get("seed", 0)),
        vecnormalize_path=vecnormalize_path,
        training=False,
    )
    env.norm_reward = False
    model = MaskablePPO.load(str(model_path), env=env)
    rollout = collect_rollout(model, env, rollout_input=rollout_input)
    rollout = add_rollout_diagnostics(rollout, config=config)
    rollout["method"] = method
    return rollout


def write_comparison_plot(
    rollouts: dict[str, pd.DataFrame],
    *,
    output_path: Path,
    config: dict[str, Any],
    title_suffix: str,
) -> None:
    methods = list(rollouts.keys())
    fig, axes = plt.subplots(
        len(methods),
        1,
        figsize=(11.0, 3.4 * max(1, len(methods))),
        sharex=True,
        sharey=True,
    )
    if len(methods) == 1:
        axes = [axes]

    capacity_per_task = max(
        infer_capacity_per_task(rollout, config=config) for rollout in rollouts.values()
    )
    # Single shared y-axis based on actual data + 20% headroom.
    upper_data = max(
        max(rollout["demand"].max(), rollout["capacity"].max(), rollout["reactive_capacity"].max())
        for rollout in rollouts.values()
    )
    y_upper = float(upper_data) * 1.20

    env_cfg = config.get("env", {})
    min_tasks = int(env_cfg.get("min_tasks", 1))

    for axis, method in zip(axes, methods, strict=True):
        rollout = rollouts[method]
        violations = int(rollout["slo_violation"].astype(bool).sum())
        task_cost = float(rollout["task_count"].sum())
        axis.plot(
            rollout["step"],
            rollout["demand"],
            linewidth=1.8,
            color="0.25",
            label="Target Demand",
        )
        axis.step(
            rollout["step"],
            rollout["capacity"],
            where="post",
            linewidth=2.0,
            color="tab:blue" if method == "replay" else "tab:red",
            label=f"Agent Capacity ({METHOD_TITLE[method]})",
        )
        axis.step(
            rollout["step"],
            rollout["reactive_capacity"],
            where="post",
            linestyle=":",
            linewidth=1.6,
            color="tab:gray",
            label="Reactive Capacity",
        )
        slo_steps = rollout[rollout["slo_violation"].astype(bool)]
        if not slo_steps.empty:
            axis.scatter(
                slo_steps["step"],
                slo_steps["demand"],
                marker="x",
                s=26,
                color="tab:red",
                linewidth=1.2,
                label="SLO Violation",
            )
        axis.set_ylim(0.0, y_upper)
        axis.set_ylabel("Demand / Capacity")
        axis.set_title(
            f"{METHOD_TITLE[method]} — {violations} SLO viols, total task-cost {task_cost:.0f}"
        )
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, loc="upper right", fontsize=8)
        axis.axhline(min_tasks * capacity_per_task, color="0.85", linewidth=0.6)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Side-by-side rollout: {title_suffix}", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
