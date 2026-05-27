"""Run continuous six-checkpoint Azure sensitivity and replay evaluation."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from calibrate_baselines import calibrate_static_p95
from run_drift_experiment import build_single_task_config, evaluate_task_a, write_calibrated_trace

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from driftscale.agents.train_ppo import train_maskable_ppo
from driftscale.agents.train_ppo_replay import train_replay_ppo
from driftscale.baselines.reactive import ReactiveAutoscaler
from driftscale.envs.reward import RewardConfig
from driftscale.traces.azure_loader import (
    AZURE_CHECKPOINT_IDS,
    load_azure_checkpoint_regimes,
)
from driftscale.traces.preprocess import (
    DemandMappingConfig,
    build_preprocessed_trace,
    write_preprocessed_trace,
)
from driftscale.utils.config import load_yaml


@dataclass(frozen=True)
class ContinuousVariantResult:
    """Summary and trajectory for one §5.5 mapping variant."""

    summary: dict[str, float | int | str]
    trajectory: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping-configs",
        nargs="+",
        default=[
            "configs/env/linear.yaml",
            "configs/env/convex.yaml",
            "configs/env/threshold.yaml",
        ],
    )
    parser.add_argument("--ppo-config", default="configs/train/ppo.yaml")
    parser.add_argument("--replay-config", default="configs/train/ppo_replay.yaml")
    parser.add_argument("--output-dir", default="results/sensitivity")
    parser.add_argument("--plot-path", default="media/continuous_forgetting.png")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=list(AZURE_CHECKPOINT_IDS))
    parser.add_argument("--vm-count", type=int, default=1000)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Refresh visual evaluation artifacts from cached sensitivity outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ppo_base = load_yaml(args.ppo_config)
    replay_base = load_yaml(args.replay_config)
    mapping_configs = [Path(config_path) for config_path in args.mapping_configs]

    if args.plot_only:
        trajectories = refresh_cached_reactive_baseline(
            output_dir=output_dir,
            mapping_configs=mapping_configs,
            ppo_base=ppo_base,
        )
        refresh_cached_summary_with_reactive(output_dir=output_dir, trajectory=trajectories)
        write_continuous_forgetting_plot(
            trajectories,
            checkpoint_ids=list(args.checkpoints),
            output_path=Path(args.plot_path),
        )
        print(f"Updated {args.plot_path}")
        return

    regimes = load_azure_checkpoint_regimes(
        raw_dir=args.raw_dir,
        checkpoint_ids=tuple(args.checkpoints),
        vm_count=args.vm_count,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )

    results = [
        run_mapping_variant(
            config_path,
            checkpoint_ids=regimes.checkpoint_ids,
            matrices=regimes.matrices,
            selected_vm_count=len(regimes.selected_vms),
            ppo_base=ppo_base,
            replay_base=replay_base,
            output_dir=output_dir,
        )
        for config_path in mapping_configs
    ]

    summary = pd.DataFrame([result.summary for result in results])
    trajectories = pd.concat([result.trajectory for result in results], ignore_index=True)
    summary_path = output_dir / "summary.csv"
    markdown_path = output_dir / "summary.md"
    trajectory_path = output_dir / "continuous_rewards.csv"
    summary.to_csv(summary_path, index=False)
    trajectories.to_csv(trajectory_path, index=False)
    write_continuous_forgetting_plot(
        trajectories,
        checkpoint_ids=regimes.checkpoint_ids,
        output_path=Path(args.plot_path),
    )

    markdown = to_markdown_table(summary)
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


def run_mapping_variant(
    config_path: Path,
    *,
    checkpoint_ids: list[int],
    matrices: list[pd.DataFrame],
    selected_vm_count: int,
    ppo_base: dict,
    replay_base: dict,
    output_dir: Path,
) -> ContinuousVariantResult:
    variant_config = load_yaml(config_path)
    variant = str(variant_config["mapping"]["variant"])
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_paths = [
        variant_dir / f"task_{checkpoint_id}_preprocessed.csv" for checkpoint_id in checkpoint_ids
    ]
    traces = [
        preprocess_task(
            variant_config,
            matrix,
            output_path,
        )
        for matrix, output_path in zip(matrices, preprocessed_paths, strict=True)
    ]
    traces = bridge_trace_boundaries(traces)
    for trace, output_path in zip(traces, preprocessed_paths, strict=True):
        write_preprocessed_trace(trace, output_path)

    calibration_cfg = variant_config.get("calibration", {})
    calibration = calibrate_static_p95(
        traces[0]["demand"].to_numpy(dtype="float32"),
        min_tasks=int(ppo_base.get("env", {}).get("min_tasks", 1)),
        max_tasks=int(ppo_base.get("env", {}).get("max_tasks", 20)),
        target_min=float(calibration_cfg.get("target_slo_min", 0.01)),
        target_max=float(calibration_cfg.get("target_slo_max", 0.05)),
    )
    scale_factor = float(calibration["scale_factor"])
    capacity_per_task = float(calibration["capacity_per_task"])
    calibrated_paths = [
        write_calibrated_trace(
            trace,
            variant_dir / f"task_{checkpoint_id}_calibrated.csv",
            scale_factor=scale_factor,
            capacity_per_task=capacity_per_task,
        )
        for checkpoint_id, trace in zip(checkpoint_ids, traces, strict=True)
    ]
    reactive_rewards = evaluate_reactive_stage_rewards(
        calibrated_paths,
        base_config=ppo_base,
        variant_config=variant_config,
        capacity_per_task=capacity_per_task,
    )

    training_cfg = variant_config.get("training", {})
    task_1_training = {
        **training_cfg,
        "total_timesteps": training_cfg.get(
            "continuous_task_1_timesteps",
            training_cfg.get("task_a_timesteps", training_cfg.get("total_timesteps", 1024)),
        ),
    }
    finetune_training = {
        **training_cfg,
        "total_timesteps": training_cfg.get(
            "continuous_finetune_timesteps",
            training_cfg.get("finetune_timesteps", training_cfg.get("total_timesteps", 1024)),
        ),
    }
    ppo_task_1 = apply_reward_overrides(
        with_training_overrides(ppo_base, task_1_training),
        variant_config,
    )
    ppo_finetune = apply_reward_overrides(
        with_training_overrides(ppo_base, finetune_training),
        variant_config,
    )
    replay_variant = apply_reward_overrides(
        with_training_overrides(replay_base, finetune_training),
        variant_config,
    )

    task_1_config = build_single_task_config(
        ppo_task_1,
        trace_path=calibrated_paths[0],
        output_dir=variant_dir / "naive_stage_1",
        capacity_per_task=capacity_per_task,
    )
    task_1_metrics = train_maskable_ppo(task_1_config)
    task_1_eval = evaluate_task_a(
        task_1_config,
        task_a_path=calibrated_paths[0],
        model_path=task_1_metrics["model_path"],
        vecnormalize_path=task_1_metrics["vecnormalize_path"],
    )

    naive_rewards = [float(task_1_eval["total_reward"])]
    replay_rewards = [float(task_1_eval["total_reward"])]
    naive_metrics = task_1_metrics
    replay_metrics = task_1_metrics

    for stage_index, checkpoint_id in enumerate(checkpoint_ids[1:], start=1):
        naive_config = build_single_task_config(
            ppo_finetune,
            trace_path=calibrated_paths[stage_index],
            output_dir=variant_dir / f"naive_stage_{checkpoint_id}",
            capacity_per_task=capacity_per_task,
            init_model_path=naive_metrics["model_path"],
            init_vecnormalize_path=naive_metrics["vecnormalize_path"],
        )
        naive_metrics = train_maskable_ppo(naive_config)
        naive_eval = evaluate_task_a(
            naive_config,
            task_a_path=calibrated_paths[0],
            model_path=naive_metrics["model_path"],
            vecnormalize_path=naive_metrics["vecnormalize_path"],
        )
        naive_rewards.append(float(naive_eval["total_reward"]))

        replay_config = build_continuous_replay_config(
            replay_variant,
            current_task_path=calibrated_paths[stage_index],
            previous_task_paths=calibrated_paths[:stage_index],
            output_dir=variant_dir / f"replay_stage_{checkpoint_id}",
            capacity_per_task=capacity_per_task,
            init_model_path=replay_metrics["model_path"],
            init_vecnormalize_path=replay_metrics["vecnormalize_path"],
            replay_mix_ratio=float(training_cfg.get("replay_mix_ratio", 0.25)),
            n_envs=int(training_cfg.get("n_envs", 4)),
        )
        replay_metrics = train_replay_ppo(replay_config)
        replay_eval = evaluate_task_a(
            replay_config,
            task_a_path=calibrated_paths[0],
            model_path=replay_metrics["model_path"],
            vecnormalize_path=replay_metrics["vecnormalize_path"],
        )
        replay_rewards.append(float(replay_eval["total_reward"]))

    initial_reward = naive_rewards[0]
    trajectory = pd.DataFrame(
        {
            "mapping": variant,
            "checkpoint": checkpoint_ids,
            "stage": list(range(1, len(checkpoint_ids) + 1)),
            "naive_task_1_reward": naive_rewards,
            "replay_task_1_reward": replay_rewards,
            "reactive_stage_reward": reactive_rewards,
            "naive_rolling_bwt": np.asarray(naive_rewards) - initial_reward,
            "replay_rolling_bwt": np.asarray(replay_rewards) - initial_reward,
        }
    )
    trajectory.to_csv(variant_dir / "continuous_rewards.csv", index=False)

    return ContinuousVariantResult(
        summary={
            "mapping": variant,
            "persistent_vm_count": selected_vm_count,
            "initial_task_1_reward": initial_reward,
            "naive_stage_125_reward": naive_rewards[-1],
            "replay_stage_125_reward": replay_rewards[-1],
            "reactive_stage_125_reward": reactive_rewards[-1],
            "naive_final_bwt": naive_rewards[-1] - initial_reward,
            "replay_final_bwt": replay_rewards[-1] - initial_reward,
            "naive_final_retention_pct": retention_percentage(initial_reward, naive_rewards[-1]),
            "replay_final_retention_pct": retention_percentage(initial_reward, replay_rewards[-1]),
        },
        trajectory=trajectory,
    )


def preprocess_task(config: dict, cpu_matrix: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    trace = build_preprocessed_trace(
        cpu_matrix,
        mapping_config=DemandMappingConfig(**config.get("mapping", {})),
    )
    write_preprocessed_trace(trace, output_path)
    return trace


def bridge_trace_boundaries(traces: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Clean near-zero checkpoint starts before PPO sees the continuous curriculum."""
    cleaned_traces: list[pd.DataFrame] = []
    previous_demand: float | None = None

    for trace in traces:
        cleaned = trace.copy()
        demand = pd.to_numeric(cleaned["demand"], errors="raise").astype("float64")
        gap_mask = np.zeros(len(cleaned), dtype=bool)

        if previous_demand is not None and previous_demand > 0.0:
            lookahead = demand.iloc[1:4]
            lookahead = lookahead[lookahead > 0.0]
            if not lookahead.empty:
                reference = min(previous_demand, float(lookahead.median()))
                gap_threshold = max(1e-6, 0.25 * reference)
                cursor = 0
                while cursor < len(demand) and float(demand.iloc[cursor]) <= gap_threshold:
                    gap_mask[cursor] = True
                    demand.iloc[cursor] = np.nan
                    cursor += 1

        cleaned["demand"] = demand.ffill().fillna(previous_demand).astype("float32")
        cleaned["boundary_gap_filled"] = gap_mask
        previous_demand = float(cleaned["demand"].iloc[-1])
        cleaned_traces.append(cleaned)

    return cleaned_traces


def evaluate_reactive_stage_rewards(
    calibrated_paths: list[Path],
    *,
    base_config: dict,
    variant_config: dict,
    capacity_per_task: float,
) -> list[float]:
    config = apply_reward_overrides(base_config, variant_config)
    env_cfg = config.get("env", {})
    reward_cfg = RewardConfig(**config.get("reward", {}))
    autoscaler = ReactiveAutoscaler(
        min_tasks=int(env_cfg.get("min_tasks", 1)),
        max_tasks=int(env_cfg.get("max_tasks", 20)),
        initial_tasks=int(env_cfg.get("initial_tasks", 1)),
        capacity_per_task=capacity_per_task,
        reward_config=reward_cfg,
    )
    rewards = []
    for trace_path in calibrated_paths:
        trace = pd.read_csv(trace_path)
        demand = trace["demand"].to_numpy(dtype=np.float32)
        rewards.append(float(autoscaler.evaluate(demand).total_reward))
    return rewards


def refresh_cached_reactive_baseline(
    *,
    output_dir: Path,
    mapping_configs: list[Path],
    ppo_base: dict,
) -> pd.DataFrame:
    trajectory_path = output_dir / "continuous_rewards.csv"
    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"{trajectory_path} does not exist; run make sensitivity-suite first"
        )
    trajectories = pd.read_csv(trajectory_path)
    frames = []
    for config_path in mapping_configs:
        variant_config = load_yaml(config_path)
        variant = str(variant_config["mapping"]["variant"])
        variant_rows = trajectories[trajectories["mapping"] == variant].copy()
        if variant_rows.empty:
            continue
        stage_paths = [
            output_dir / variant / f"task_{int(checkpoint)}_calibrated.csv"
            for checkpoint in variant_rows["checkpoint"]
        ]
        first_stage = pd.read_csv(stage_paths[0])
        capacity_per_task = float(first_stage["capacity_per_task"].iloc[0])
        variant_rows["reactive_stage_reward"] = evaluate_reactive_stage_rewards(
            stage_paths,
            base_config=ppo_base,
            variant_config=variant_config,
            capacity_per_task=capacity_per_task,
        )
        frames.append(variant_rows)

    if not frames:
        raise ValueError("no cached mapping trajectories matched the configured variants")
    refreshed = pd.concat(frames, ignore_index=True)
    refreshed.to_csv(trajectory_path, index=False)
    return refreshed


def refresh_cached_summary_with_reactive(*, output_dir: Path, trajectory: pd.DataFrame) -> None:
    summary_path = output_dir / "summary.csv"
    markdown_path = output_dir / "summary.md"
    if not summary_path.exists():
        return

    summary = pd.read_csv(summary_path)
    final_reactive = (
        trajectory.sort_values("stage")
        .groupby("mapping", as_index=False)
        .tail(1)[["mapping", "reactive_stage_reward"]]
        .rename(columns={"reactive_stage_reward": "reactive_stage_125_reward"})
    )
    summary = summary.drop(columns=["reactive_stage_125_reward"], errors="ignore").merge(
        final_reactive,
        on="mapping",
        how="left",
    )
    summary.to_csv(summary_path, index=False)
    markdown_path.write_text(to_markdown_table(summary) + "\n", encoding="utf-8")


def build_continuous_replay_config(
    base_config: dict,
    *,
    current_task_path: Path,
    previous_task_paths: list[Path],
    output_dir: Path,
    capacity_per_task: float,
    init_model_path: str | Path,
    init_vecnormalize_path: str | Path,
    replay_mix_ratio: float,
    n_envs: int,
) -> dict:
    config = copy.deepcopy(base_config)
    config["current_task"] = {"path": str(current_task_path), "demand_column": "demand"}
    config["previous_tasks"] = [
        {"path": str(task_path), "demand_column": "demand"} for task_path in previous_task_paths
    ]
    config["eval_task"] = {"path": str(previous_task_paths[0]), "demand_column": "demand"}
    config["output_dir"] = str(output_dir)
    config["init_model_path"] = str(init_model_path)
    config["init_vecnormalize_path"] = str(init_vecnormalize_path)
    config.setdefault("replay", {})["replay_mix_ratio"] = replay_mix_ratio
    config["replay"]["n_envs"] = n_envs
    config.setdefault("env", {})["capacity_per_task"] = capacity_per_task
    config.setdefault("env", {}).setdefault("initial_tasks", 1)
    config.setdefault("eval", {})["deterministic"] = True
    return config


def with_training_overrides(base_config: dict, training_cfg: dict) -> dict:
    config = copy.deepcopy(base_config)
    if "total_timesteps" in training_cfg:
        config.setdefault("ppo", {})["total_timesteps"] = int(training_cfg["total_timesteps"])
    if "n_envs" in training_cfg:
        config.setdefault("replay", {})["n_envs"] = int(training_cfg["n_envs"])
    return config


def apply_reward_overrides(config: dict, variant_config: dict) -> dict:
    updated = copy.deepcopy(config)
    if "reward" in variant_config:
        updated.setdefault("reward", {}).update(variant_config["reward"])
    if "env" in variant_config:
        updated.setdefault("env", {}).update(variant_config["env"])
    return updated


def retention_percentage(initial_reward: float, final_reward: float) -> float:
    denominator = max(abs(initial_reward), 1e-6)
    drop = max(0.0, initial_reward - final_reward)
    return float(np.clip(100.0 * (1.0 - drop / denominator), 0.0, 100.0))


def write_continuous_forgetting_plot(
    trajectory: pd.DataFrame,
    *,
    checkpoint_ids: list[int],
    output_path: Path,
) -> None:
    plot_columns = ["naive_task_1_reward", "replay_task_1_reward"]
    if "reactive_stage_reward" in trajectory.columns:
        plot_columns.append("reactive_stage_reward")
    grouped = trajectory.groupby("checkpoint", sort=False)[plot_columns].mean()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    axis.plot(
        checkpoint_ids,
        grouped["naive_task_1_reward"].to_numpy(),
        marker="o",
        linewidth=2.0,
        label="Naive Fine-Tuning",
    )
    axis.plot(
        checkpoint_ids,
        grouped["replay_task_1_reward"].to_numpy(),
        marker="o",
        linewidth=2.0,
        label="PPO + Replay",
    )
    if "reactive_stage_reward" in grouped.columns:
        axis.plot(
            checkpoint_ids,
            grouped["reactive_stage_reward"].to_numpy(),
            linestyle=":",
            color="0.25",
            linewidth=2.25,
            label="Reactive Baseline",
        )
    axis.set_xlabel("Training Stage")
    axis.set_ylabel("Task 1 Reward")
    axis.set_title("Continuous Forgetting Across Azure Checkpoints")
    axis.set_xticks(checkpoint_ids)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def to_markdown_table(frame: pd.DataFrame) -> str:
    headers = [
        "Mapping",
        "VMs",
        "Initial Task 1 Reward",
        "Naive Stage 125 Reward",
        "Replay Stage 125 Reward",
        "Reactive Stage 125 Reward",
        "Naive Final BWT",
        "Replay Final BWT",
        "Naive Retention",
        "Replay Retention",
    ]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.to_dict("records"):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["mapping"]),
                    f"{int(row['persistent_vm_count'])}",
                    f"{float(row['initial_task_1_reward']):.3f}",
                    f"{float(row['naive_stage_125_reward']):.3f}",
                    f"{float(row['replay_stage_125_reward']):.3f}",
                    f"{float(row['reactive_stage_125_reward']):.3f}",
                    f"{float(row['naive_final_bwt']):.3f}",
                    f"{float(row['replay_final_bwt']):.3f}",
                    f"{float(row['naive_final_retention_pct']):.1f}%",
                    f"{float(row['replay_final_retention_pct']):.1f}%",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
