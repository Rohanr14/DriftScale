"""Plot a replay-policy rollout against cached continuous Azure demand."""

from __future__ import annotations

import argparse
import copy
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
from driftscale.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default="linear")
    parser.add_argument("--stage", type=int, default=125)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--rollout-mode",
        choices=("curriculum", "stage"),
        default="curriculum",
        help="Use all six cached checkpoints or only the final stage.",
    )
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[1, 25, 50, 75, 100, 125])
    parser.add_argument("--sensitivity-dir", default="results/sensitivity")
    parser.add_argument("--train-config", default="configs/train/ppo_replay.yaml")
    parser.add_argument("--env-config")
    parser.add_argument("--model-path")
    parser.add_argument("--vecnormalize-path")
    parser.add_argument("--trace-path")
    parser.add_argument("--csv-path")
    parser.add_argument("--output-path", default="media/episode_rollout.png")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the rollout PNG from the cached rollout CSV without loading model artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    if args.plot_only:
        rollout = pd.read_csv(paths["csv"])
        write_rollout_plot(
            rollout,
            output_path=paths["output"],
            title=rollout_title(args),
            config=rollout_config(
                train_config_path=paths["train_config"],
                env_config_path=paths["env_config"],
                capacity_trace_path=paths["trace_paths"][0],
            ),
        )
        print(f"Updated rollout plot from {paths['csv']}")
        return

    rollout_input = load_rollout_input(
        trace_paths=paths["trace_paths"],
        checkpoints=args.checkpoints,
    )
    config = rollout_config(
        train_config_path=paths["train_config"],
        env_config_path=paths["env_config"],
        capacity_trace_path=paths["trace_paths"][0],
    )
    env = build_vecnormalize_env(
        config,
        demand=rollout_input["demand"].to_numpy(dtype=np.float32),
        seed=int(config.get("seed", 0)),
        vecnormalize_path=paths["vecnormalize"],
        training=False,
    )
    env.norm_reward = False
    model = MaskablePPO.load(str(paths["model"]), env=env)

    rollout = collect_rollout(model, env, rollout_input=rollout_input)
    rollout = add_rollout_diagnostics(rollout, config=config)
    csv_path = paths["csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rollout.to_csv(csv_path, index=False)
    write_rollout_plot(
        rollout,
        output_path=paths["output"],
        title=rollout_title(args),
        config=config,
    )
    print(f"Saved rollout CSV to {csv_path}")
    print(f"Saved rollout plot to {paths['output']}")


def resolve_paths(args: argparse.Namespace) -> dict[str, Any]:
    sensitivity_dir = Path(args.sensitivity_dir)
    mapping_dir = sensitivity_dir / args.mapping
    seeded_stage_dir = mapping_dir / f"seed_{args.seed}" / f"replay_stage_{args.stage}"
    legacy_stage_dir = mapping_dir / f"replay_stage_{args.stage}"
    stage_dir = seeded_stage_dir if seeded_stage_dir.exists() else legacy_stage_dir
    if args.trace_path:
        trace_paths = [Path(args.trace_path)]
    elif args.rollout_mode == "stage":
        trace_paths = [mapping_dir / f"task_{args.stage}_calibrated.csv"]
    else:
        trace_paths = [
            mapping_dir / f"task_{checkpoint}_calibrated.csv" for checkpoint in args.checkpoints
        ]
    default_csv_name = (
        f"episode_rollout_stage_{args.stage}.csv"
        if args.rollout_mode == "stage"
        else "episode_rollout_curriculum.csv"
    )
    paths = {
        "train_config": Path(args.train_config),
        "env_config": Path(args.env_config or f"configs/env/{args.mapping}.yaml"),
        "model": Path(args.model_path or stage_dir / "model.zip"),
        "vecnormalize": Path(args.vecnormalize_path or stage_dir / "vecnormalize.pkl"),
        "trace_paths": trace_paths,
        "csv": Path(args.csv_path or mapping_dir / default_csv_name),
        "output": Path(args.output_path),
    }
    for label, path in paths.items():
        if label in {"csv", "output"}:
            continue
        if args.plot_only and label in {"model", "vecnormalize"}:
            continue
        if label == "trace_paths":
            for trace_path in path:
                if not trace_path.exists():
                    raise FileNotFoundError(
                        f"{trace_path} does not exist; run make sensitivity-suite first"
                    )
            continue
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist; run make sensitivity-suite first")
    return paths


def load_rollout_input(*, trace_paths: list[Path], checkpoints: list[int]) -> pd.DataFrame:
    frames = []
    for index, trace_path in enumerate(trace_paths):
        trace = pd.read_csv(trace_path)
        checkpoint = checkpoints[index] if index < len(checkpoints) else -1
        frames.append(
            pd.DataFrame(
                {
                    "demand": trace["demand"].astype("float32"),
                    "checkpoint": checkpoint,
                    "checkpoint_step": np.arange(len(trace), dtype=np.int32),
                }
            )
        )
    rollout_input = pd.concat(frames, ignore_index=True)
    rollout_input = bridge_boundary_gaps(rollout_input)
    rollout_input["global_step"] = np.arange(len(rollout_input), dtype=np.int32)
    return rollout_input


def bridge_boundary_gaps(rollout_input: pd.DataFrame) -> pd.DataFrame:
    """Treat zero-filled checkpoint boundary rows as missing, then ffill across joins."""
    cleaned = rollout_input.copy()
    original_demand = pd.to_numeric(cleaned["demand"], errors="raise").astype("float32")
    demand = original_demand.astype("float64").copy()
    gap_mask = pd.Series(False, index=cleaned.index)

    boundary_starts = cleaned.groupby("checkpoint", sort=False).head(1).index[1:]
    for start in boundary_starts:
        previous_value = float(demand.iloc[start - 1])
        lookahead = demand.iloc[start + 1 : start + 4]
        lookahead = lookahead[lookahead > 0.0]
        if lookahead.empty or previous_value <= 0.0:
            continue

        reference = min(previous_value, float(lookahead.median()))
        gap_threshold = max(1e-6, 0.25 * reference)
        cursor = int(start)
        while cursor < len(demand) and float(demand.iloc[cursor]) <= gap_threshold:
            gap_mask.iloc[cursor] = True
            demand.iloc[cursor] = np.nan
            cursor += 1

    cleaned["demand"] = demand.ffill().bfill().astype("float32")
    cleaned["boundary_gap_filled"] = gap_mask.to_numpy(dtype=bool)
    cleaned["raw_demand"] = original_demand
    return cleaned


def rollout_config(
    *,
    train_config_path: Path,
    env_config_path: Path,
    capacity_trace_path: Path,
) -> dict[str, Any]:
    config = load_yaml(train_config_path)
    env_config = load_yaml(env_config_path)
    updated = copy.deepcopy(config)
    if "reward" in env_config:
        updated.setdefault("reward", {}).update(env_config["reward"])
    if "env" in env_config:
        updated.setdefault("env", {}).update(env_config["env"])

    trace = pd.read_csv(capacity_trace_path)
    capacity_per_task = float(
        trace["capacity_per_task"].iloc[0]
        if "capacity_per_task" in trace.columns
        else updated.get("env", {}).get("capacity_per_task", 1.0)
    )
    updated.setdefault("env", {})["capacity_per_task"] = capacity_per_task
    updated["env"].setdefault("initial_tasks", 1)
    updated.setdefault("eval", {})["deterministic"] = True
    return updated


def collect_rollout(
    model: MaskablePPO,
    env,
    *,
    rollout_input: pd.DataFrame,
) -> pd.DataFrame:
    obs = env.reset()
    done = np.array([False])
    records = []

    while not bool(done[0]):
        masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        obs, _rewards, done, infos = env.step(action)
        info = infos[0]
        step = int(info["step"])
        metadata = rollout_input.iloc[step]
        records.append(
            {
                "step": step,
                "checkpoint": int(metadata["checkpoint"]),
                "checkpoint_step": int(metadata["checkpoint_step"]),
                "boundary_gap_filled": bool(metadata["boundary_gap_filled"]),
                "raw_demand": float(metadata["raw_demand"]),
                "demand": float(info["demand"]),
                "capacity": float(info["capacity"]),
                "task_count": int(info["task_count"]),
            }
        )

    return pd.DataFrame(records)


def add_rollout_diagnostics(rollout: pd.DataFrame, *, config: dict[str, Any]) -> pd.DataFrame:
    updated = rollout.copy()
    env_cfg = config.get("env", {})
    capacity_per_task = infer_capacity_per_task(updated, config=config)
    reactive_tasks = reactive_task_trace(
        updated["demand"].to_numpy(dtype=np.float32),
        min_tasks=int(env_cfg.get("min_tasks", 1)),
        max_tasks=int(env_cfg.get("max_tasks", 20)),
        initial_tasks=int(env_cfg.get("initial_tasks", 1)),
        capacity_per_task=capacity_per_task,
    )
    updated["capacity_per_task"] = capacity_per_task
    updated["slo_violation"] = updated["demand"] > updated["capacity"]
    updated["reactive_task_count"] = reactive_tasks
    updated["reactive_capacity"] = reactive_tasks.astype(np.float32) * capacity_per_task
    return updated


def infer_capacity_per_task(rollout: pd.DataFrame, *, config: dict[str, Any]) -> float:
    if "capacity_per_task" in rollout.columns and not rollout["capacity_per_task"].isna().all():
        return float(rollout["capacity_per_task"].dropna().iloc[0])
    if "task_count" in rollout.columns and "capacity" in rollout.columns:
        ratios = rollout["capacity"] / rollout["task_count"].replace(0, np.nan)
        ratios = ratios.dropna()
        if not ratios.empty:
            return float(ratios.median())
    return float(config.get("env", {}).get("capacity_per_task", 1.0))


def reactive_task_trace(
    demand: np.ndarray,
    *,
    min_tasks: int,
    max_tasks: int,
    initial_tasks: int,
    capacity_per_task: float,
) -> np.ndarray:
    task_count = int(np.clip(initial_tasks, min_tasks, max_tasks))
    high_count = 0
    low_count = 0
    task_history = []
    for demand_t in demand:
        observed_cpu = float(demand_t / max(task_count * capacity_per_task, 1e-8))
        if observed_cpu > 0.70:
            high_count += 1
            low_count = 0
        elif observed_cpu < 0.30:
            low_count += 1
            high_count = 0
        else:
            high_count = 0
            low_count = 0

        if high_count >= 1:
            task_count = min(max_tasks, task_count + 2)
            high_count = 0
        elif low_count >= 6:
            task_count = max(min_tasks, task_count - 1)
            low_count = 0
        task_history.append(task_count)
    return np.asarray(task_history, dtype=np.int32)


def rollout_title(args: argparse.Namespace) -> str:
    if args.rollout_mode == "stage":
        return f"Stage {args.stage} Replay Rollout ({args.mapping}, seed {args.seed})"
    return f"Six-Checkpoint Replay Rollout ({args.mapping}, seed {args.seed})"


def write_rollout_plot(
    rollout: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    config: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "reactive_capacity" not in rollout.columns or "slo_violation" not in rollout.columns:
        rollout = add_rollout_diagnostics(rollout, config=config)
    env_cfg = config.get("env", {})
    capacity_per_task = infer_capacity_per_task(rollout, config=config)
    min_tasks = int(env_cfg.get("min_tasks", 1))
    max_tasks = int(env_cfg.get("max_tasks", int(rollout["task_count"].max())))

    figure, axis = plt.subplots(figsize=(10.0, 5.2))
    axis.plot(
        rollout["step"],
        rollout["demand"],
        linewidth=2.0,
        label="Target Demand",
    )
    axis.step(
        rollout["step"],
        rollout["capacity"],
        where="post",
        linewidth=2.0,
        label="Agent Capacity",
    )
    axis.step(
        rollout["step"],
        rollout["reactive_capacity"],
        where="post",
        linestyle=":",
        linewidth=2.0,
        color="0.25",
        label="Reactive Capacity",
    )
    slo_steps = rollout[rollout["slo_violation"].astype(bool)]
    if not slo_steps.empty:
        axis.scatter(
            slo_steps["step"],
            slo_steps["demand"],
            marker="x",
            s=28,
            color="tab:red",
            linewidth=1.2,
            label="SLO Violation",
        )
    task_axis = axis.twinx()
    task_axis.step(
        rollout["step"],
        rollout["task_count"],
        where="post",
        color="tab:green",
        linewidth=1.3,
        alpha=0.75,
        label="Agent Tasks",
    )
    axis.set_xlabel("Timestep")
    axis.set_ylabel("Demand / Capacity")
    task_axis.set_ylabel("Task Count")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    upper_capacity = max_tasks * capacity_per_task
    observed_upper = max(float(rollout["demand"].max()), float(rollout["capacity"].max()))
    axis.set_ylim(0.0, max(upper_capacity, observed_upper) * 1.08)
    task_axis.set_ylim(max(0, min_tasks - 1), max_tasks + 1)
    lines, labels = axis.get_legend_handles_labels()
    task_lines, task_labels = task_axis.get_legend_handles_labels()
    axis.legend(lines + task_lines, labels + task_labels, frameon=False, loc="upper right")
    if "checkpoint" in rollout.columns and rollout["checkpoint"].nunique() > 1:
        boundaries = rollout.groupby("checkpoint", sort=False)["step"].min().iloc[1:]
        top = max(float(rollout["capacity"].max()), float(rollout["demand"].max()))
        for boundary in boundaries:
            axis.axvline(int(boundary), color="0.75", linewidth=0.8, linestyle="--")
        for checkpoint, start in rollout.groupby("checkpoint", sort=False)["step"].min().items():
            axis.text(
                int(start),
                top,
                str(int(checkpoint)),
                fontsize=8,
                color="0.35",
                va="bottom",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
