"""Run continuous six-checkpoint Azure sensitivity and replay evaluation."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from driftscale.eval.drift import demand_distribution_diagnostics, drift_diagnostic_passed
from driftscale.traces.azure_loader import (
    AZURE_CHECKPOINT_IDS,
    VmSelectionStrategy,
    load_azure_checkpoint_regimes,
)
from driftscale.traces.preprocess import (
    DemandMappingConfig,
    build_preprocessed_trace,
    write_preprocessed_trace,
)
from driftscale.traces.regimes import WorkloadRegime
from driftscale.traces.synthetic import generate_synthetic_episode
from driftscale.utils.config import load_yaml


@dataclass(frozen=True)
class ContinuousVariantResult:
    """Summary and trajectory for one §5.5 mapping variant."""

    summary: dict[str, float | int | str]
    trajectory: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class Curriculum:
    """Loaded checkpoint matrices and provenance for the continuous experiment."""

    checkpoint_ids: list[int]
    matrices: list[pd.DataFrame]
    selected_vm_count: int
    selection_strategy: str
    source: str


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
    parser.add_argument("--vm-count", type=int, default=32)
    parser.add_argument(
        "--vm-selection-strategy",
        choices=[strategy.value for strategy in VmSelectionStrategy],
        default=VmSelectionStrategy.PER_CHECKPOINT_DENSE.value,
        help=(
            "Azure VM cohort selection. per_checkpoint_dense preserves shard-level drift; "
            "persistent_dense reproduces the old all-shard intersection."
        ),
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--seed-count",
        type=int,
        default=3,
        help=(
            "Number of consecutive seeds to run, starting at the PPO config seed unless "
            "--seeds is supplied. Default 3; runtime scales roughly linearly."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--task-1-timesteps",
        type=int,
        help="Override continuous Task-1 PPO timesteps for every mapping.",
    )
    parser.add_argument(
        "--finetune-timesteps",
        type=int,
        help="Override each post-drift fine-tuning stage's PPO timesteps.",
    )
    parser.add_argument("--replay-mix-ratio", type=float)
    parser.add_argument(
        "--curriculum-source",
        choices=("auto", "azure", "synthetic"),
        default="auto",
        help=(
            "auto uses real Azure checkpoints when the drift diagnostic passes and falls back "
            "to labeled synthetic regimes otherwise. Full default run is about 10-30 minutes "
            "on a laptop with 3 seeds and the bundled PPO budgets."
        ),
    )
    parser.add_argument("--min-drift-smd", type=float, default=0.5)
    parser.add_argument("--min-drift-ks", type=float, default=0.30)
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

    seeds = resolve_seeds(args, ppo_base)
    curriculum = load_curriculum(
        args,
        seed=int(ppo_base.get("seed", 0)),
    )
    validation_diagnostics = matrix_demand_diagnostics(
        curriculum.matrices,
        checkpoint_ids=curriculum.checkpoint_ids,
        mapping="linear_raw",
        curriculum_source=curriculum.source,
        vm_selection_strategy=curriculum.selection_strategy,
        selected_vm_count=curriculum.selected_vm_count,
        min_drift_smd=args.min_drift_smd,
        min_drift_ks=args.min_drift_ks,
    )
    if curriculum.source == "azure" and not drift_diagnostic_passed(validation_diagnostics):
        if args.curriculum_source == "azure":
            raise ValueError(
                "Azure checkpoint drift diagnostic failed; try a lower --vm-count, a different "
                "--vm-selection-strategy, or --curriculum-source synthetic."
            )
        print(
            "Azure checkpoint drift diagnostic failed; falling back to labeled synthetic regimes."
        )
        curriculum = load_synthetic_curriculum(args, seed=int(ppo_base.get("seed", 0)))
        validation_diagnostics = matrix_demand_diagnostics(
            curriculum.matrices,
            checkpoint_ids=curriculum.checkpoint_ids,
            mapping="linear_raw",
            curriculum_source=curriculum.source,
            vm_selection_strategy=curriculum.selection_strategy,
            selected_vm_count=curriculum.selected_vm_count,
            min_drift_smd=args.min_drift_smd,
            min_drift_ks=args.min_drift_ks,
        )
    print("Demand drift diagnostic (linear raw aggregate):")
    print(format_diagnostic_table(validation_diagnostics))

    results = [
        run_mapping_variant(
            config_path,
            checkpoint_ids=curriculum.checkpoint_ids,
            matrices=curriculum.matrices,
            selected_vm_count=curriculum.selected_vm_count,
            vm_selection_strategy=curriculum.selection_strategy,
            curriculum_source=curriculum.source,
            seeds=seeds,
            ppo_base=ppo_base,
            replay_base=replay_base,
            output_dir=output_dir,
            task_1_timesteps=args.task_1_timesteps,
            finetune_timesteps=args.finetune_timesteps,
            replay_mix_ratio=args.replay_mix_ratio,
            min_drift_smd=args.min_drift_smd,
            min_drift_ks=args.min_drift_ks,
        )
        for config_path in mapping_configs
    ]

    summary = pd.DataFrame([result.summary for result in results])
    trajectories = pd.concat([result.trajectory for result in results], ignore_index=True)
    diagnostics = pd.concat(
        [validation_diagnostics, *[result.diagnostics for result in results]],
        ignore_index=True,
    )
    summary_path = output_dir / "summary.csv"
    markdown_path = output_dir / "summary.md"
    trajectory_path = output_dir / "continuous_rewards.csv"
    diagnostics_path = output_dir / "demand_diagnostics.csv"
    summary.to_csv(summary_path, index=False)
    trajectories.to_csv(trajectory_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    write_continuous_forgetting_plot(
        trajectories,
        checkpoint_ids=curriculum.checkpoint_ids,
        output_path=Path(args.plot_path),
    )

    markdown = to_markdown_table(summary)
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


def resolve_seeds(args: argparse.Namespace, ppo_base: dict[str, Any]) -> list[int]:
    if args.seeds:
        return list(dict.fromkeys(args.seeds))
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    start_seed = int(ppo_base.get("seed", 0))
    return [start_seed + offset for offset in range(args.seed_count)]


def load_curriculum(args: argparse.Namespace, *, seed: int) -> Curriculum:
    if args.curriculum_source == "synthetic":
        return load_synthetic_curriculum(args, seed=seed)

    regimes = load_azure_checkpoint_regimes(
        raw_dir=args.raw_dir,
        checkpoint_ids=tuple(args.checkpoints),
        vm_count=args.vm_count,
        selection_strategy=args.vm_selection_strategy,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )
    selected_counts = [len(selected) for selected in regimes.selected_vms_by_checkpoint]
    return Curriculum(
        checkpoint_ids=regimes.checkpoint_ids,
        matrices=regimes.matrices,
        selected_vm_count=int(min(selected_counts)),
        selection_strategy=regimes.selection_strategy,
        source="azure",
    )


def load_synthetic_curriculum(args: argparse.Namespace, *, seed: int) -> Curriculum:
    checkpoint_ids = list(args.checkpoints)
    regime_sequence = [
        WorkloadRegime.STABLE_DIURNAL,
        WorkloadRegime.STABLE_DIURNAL,
        WorkloadRegime.BURSTY,
        WorkloadRegime.BURSTY,
        WorkloadRegime.HIGH_SUSTAINED,
        WorkloadRegime.HIGH_SUSTAINED,
    ]
    matrices = []
    for index, checkpoint_id in enumerate(checkpoint_ids):
        regime = regime_sequence[min(index, len(regime_sequence) - 1)]
        episode = generate_synthetic_episode(regime=regime, length=288, seed=seed + index)
        matrices.append(synthetic_episode_to_cpu_matrix(episode.demand, vm_count=args.vm_count))
        matrices[-1].index.name = f"synthetic_checkpoint_{checkpoint_id}"
    return Curriculum(
        checkpoint_ids=checkpoint_ids,
        matrices=matrices,
        selected_vm_count=args.vm_count,
        selection_strategy="synthetic_regime",
        source="synthetic",
    )


def synthetic_episode_to_cpu_matrix(demand: np.ndarray, *, vm_count: int) -> pd.DataFrame:
    if vm_count <= 0:
        raise ValueError("vm_count must be positive")
    weights = np.linspace(1.8, 0.4, vm_count, dtype=np.float32)
    weights = weights / weights.sum()
    cpu = np.minimum(demand[:, None] * weights[None, :], 1.0)
    return pd.DataFrame(cpu, columns=[f"synthetic-vm-{index:03d}" for index in range(vm_count)])


def run_mapping_variant(
    config_path: Path,
    *,
    checkpoint_ids: list[int],
    matrices: list[pd.DataFrame],
    selected_vm_count: int,
    vm_selection_strategy: str,
    curriculum_source: str,
    seeds: list[int],
    ppo_base: dict,
    replay_base: dict,
    output_dir: Path,
    task_1_timesteps: int | None,
    finetune_timesteps: int | None,
    replay_mix_ratio: float | None,
    min_drift_smd: float,
    min_drift_ks: float,
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

    diagnostics = demand_diagnostics_from_traces(
        traces,
        checkpoint_ids=checkpoint_ids,
        mapping=variant,
        curriculum_source=curriculum_source,
        vm_selection_strategy=vm_selection_strategy,
        selected_vm_count=selected_vm_count,
        min_drift_smd=min_drift_smd,
        min_drift_ks=min_drift_ks,
    )
    print(f"Demand drift diagnostic ({variant} mapping):")
    print(format_diagnostic_table(diagnostics))

    trajectories = [
        run_mapping_seed(
            variant=variant,
            variant_config=variant_config,
            variant_dir=variant_dir,
            checkpoint_ids=checkpoint_ids,
            calibrated_paths=calibrated_paths,
            reactive_rewards=reactive_rewards,
            capacity_per_task=capacity_per_task,
            ppo_base=ppo_base,
            replay_base=replay_base,
            seed=seed,
            task_1_timesteps=task_1_timesteps,
            finetune_timesteps=finetune_timesteps,
            replay_mix_ratio=replay_mix_ratio,
        )
        for seed in seeds
    ]
    trajectory = pd.concat(trajectories, ignore_index=True)
    trajectory.to_csv(variant_dir / "continuous_rewards.csv", index=False)

    return ContinuousVariantResult(
        summary=summarize_variant(
            trajectory,
            mapping=variant,
            selected_vm_count=selected_vm_count,
            vm_selection_strategy=vm_selection_strategy,
            curriculum_source=curriculum_source,
            seeds=seeds,
        ),
        trajectory=trajectory,
        diagnostics=diagnostics,
    )


def run_mapping_seed(
    *,
    variant: str,
    variant_config: dict[str, Any],
    variant_dir: Path,
    checkpoint_ids: list[int],
    calibrated_paths: list[Path],
    reactive_rewards: list[float],
    capacity_per_task: float,
    ppo_base: dict[str, Any],
    replay_base: dict[str, Any],
    seed: int,
    task_1_timesteps: int | None,
    finetune_timesteps: int | None,
    replay_mix_ratio: float | None,
) -> pd.DataFrame:
    training_cfg = dict(variant_config.get("training", {}))
    if task_1_timesteps is not None:
        training_cfg["continuous_task_1_timesteps"] = int(task_1_timesteps)
    if finetune_timesteps is not None:
        training_cfg["continuous_finetune_timesteps"] = int(finetune_timesteps)
    if replay_mix_ratio is not None:
        training_cfg["replay_mix_ratio"] = float(replay_mix_ratio)

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
    ppo_task_1 = with_seed(
        apply_reward_overrides(with_training_overrides(ppo_base, task_1_training), variant_config),
        seed=seed,
    )
    ppo_finetune = with_seed(
        apply_reward_overrides(
            with_training_overrides(ppo_base, finetune_training),
            variant_config,
        ),
        seed=seed,
    )
    replay_variant = with_seed(
        apply_reward_overrides(
            with_training_overrides(replay_base, finetune_training),
            variant_config,
        ),
        seed=seed,
    )

    seed_dir = variant_dir / f"seed_{seed}"
    task_1_config = build_single_task_config(
        ppo_task_1,
        trace_path=calibrated_paths[0],
        output_dir=seed_dir / "naive_stage_1",
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
            output_dir=seed_dir / f"naive_stage_{checkpoint_id}",
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
            output_dir=seed_dir / f"replay_stage_{checkpoint_id}",
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
    naive_bwt = np.asarray(naive_rewards, dtype=np.float64) - initial_reward
    replay_bwt = np.asarray(replay_rewards, dtype=np.float64) - initial_reward
    trajectory = pd.DataFrame(
        {
            "mapping": variant,
            "seed": seed,
            "checkpoint": checkpoint_ids,
            "stage": list(range(1, len(checkpoint_ids) + 1)),
            "naive_task_1_reward": naive_rewards,
            "replay_task_1_reward": replay_rewards,
            "reactive_stage_reward": reactive_rewards,
            "naive_rolling_bwt": naive_bwt,
            "replay_rolling_bwt": replay_bwt,
            "naive_forgetting": np.maximum(0.0, -naive_bwt),
            "replay_forgetting": np.maximum(0.0, -replay_bwt),
        }
    )
    trajectory.to_csv(seed_dir / "continuous_rewards.csv", index=False)
    return trajectory


def summarize_variant(
    trajectory: pd.DataFrame,
    *,
    mapping: str,
    selected_vm_count: int,
    vm_selection_strategy: str,
    curriculum_source: str,
    seeds: list[int],
) -> dict[str, float | int | str]:
    final_stage = int(trajectory["stage"].max())
    final_rows = trajectory[trajectory["stage"] == final_stage]
    summary: dict[str, float | int | str] = {
        "mapping": mapping,
        "curriculum_source": curriculum_source,
        "vm_selection_strategy": vm_selection_strategy,
        "selected_vm_count": selected_vm_count,
        "seed_count": len(seeds),
        "seeds": ",".join(str(seed) for seed in seeds),
        "final_checkpoint": int(final_rows["checkpoint"].iloc[0]),
    }
    metric_columns = [
        "initial_task_1_reward",
        "naive_stage_final_reward",
        "replay_stage_final_reward",
        "reactive_stage_final_reward",
        "naive_final_bwt",
        "replay_final_bwt",
        "naive_final_forgetting",
        "replay_final_forgetting",
    ]
    metric_values = pd.DataFrame(
        {
            "initial_task_1_reward": (
                trajectory[trajectory["stage"] == 1]
                .sort_values("seed")["naive_task_1_reward"]
                .to_numpy()
            ),
            "naive_stage_final_reward": final_rows.sort_values("seed")[
                "naive_task_1_reward"
            ].to_numpy(),
            "replay_stage_final_reward": final_rows.sort_values("seed")[
                "replay_task_1_reward"
            ].to_numpy(),
            "reactive_stage_final_reward": final_rows.sort_values("seed")[
                "reactive_stage_reward"
            ].to_numpy(),
            "naive_final_bwt": final_rows.sort_values("seed")["naive_rolling_bwt"].to_numpy(),
            "replay_final_bwt": final_rows.sort_values("seed")["replay_rolling_bwt"].to_numpy(),
            "naive_final_forgetting": final_rows.sort_values("seed")["naive_forgetting"].to_numpy(),
            "replay_final_forgetting": final_rows.sort_values("seed")[
                "replay_forgetting"
            ].to_numpy(),
        }
    )
    for column in metric_columns:
        summary[f"{column}_mean"] = float(metric_values[column].mean())
        std = float(metric_values[column].std(ddof=1))
        summary[f"{column}_std"] = 0.0 if np.isnan(std) else std
    return summary


def matrix_demand_diagnostics(
    matrices: list[pd.DataFrame],
    *,
    checkpoint_ids: list[int],
    mapping: str,
    curriculum_source: str,
    vm_selection_strategy: str,
    selected_vm_count: int,
    min_drift_smd: float,
    min_drift_ks: float,
) -> pd.DataFrame:
    demands = [matrix.sum(axis=1).to_numpy(dtype=np.float64) for matrix in matrices]
    return demand_distribution_diagnostics(
        demands,
        checkpoint_ids=checkpoint_ids,
        mapping=mapping,
        curriculum_source=curriculum_source,
        vm_selection_strategy=vm_selection_strategy,
        selected_vm_count=selected_vm_count,
        min_drift_smd=min_drift_smd,
        min_drift_ks=min_drift_ks,
    )


def demand_diagnostics_from_traces(
    traces: list[pd.DataFrame],
    *,
    checkpoint_ids: list[int],
    mapping: str,
    curriculum_source: str,
    vm_selection_strategy: str,
    selected_vm_count: int,
    min_drift_smd: float,
    min_drift_ks: float,
) -> pd.DataFrame:
    demands = [trace["demand"].to_numpy(dtype=np.float64) for trace in traces]
    return demand_distribution_diagnostics(
        demands,
        checkpoint_ids=checkpoint_ids,
        mapping=mapping,
        curriculum_source=curriculum_source,
        vm_selection_strategy=vm_selection_strategy,
        selected_vm_count=selected_vm_count,
        min_drift_smd=min_drift_smd,
        min_drift_ks=min_drift_ks,
    )


def format_diagnostic_table(diagnostics: pd.DataFrame) -> str:
    columns = [
        "checkpoint",
        "mean",
        "std",
        "p95",
        "smd_vs_checkpoint_1",
        "ks_vs_checkpoint_1",
        "measurable_shift",
    ]
    return diagnostics[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}")


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
        checkpoints = variant_rows["checkpoint"].drop_duplicates().astype(int).tolist()
        stage_paths = [
            output_dir / variant / f"task_{int(checkpoint)}_calibrated.csv"
            for checkpoint in checkpoints
        ]
        first_stage = pd.read_csv(stage_paths[0])
        capacity_per_task = float(first_stage["capacity_per_task"].iloc[0])
        reactive_rewards = evaluate_reactive_stage_rewards(
            stage_paths,
            base_config=ppo_base,
            variant_config=variant_config,
            capacity_per_task=capacity_per_task,
        )
        reward_by_checkpoint = dict(zip(checkpoints, reactive_rewards, strict=True))
        variant_rows["reactive_stage_reward"] = variant_rows["checkpoint"].map(
            reward_by_checkpoint
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
    final_stage = trajectory["stage"].max()
    final_reactive = (
        trajectory[trajectory["stage"] == final_stage]
        .groupby("mapping")["reactive_stage_reward"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "reactive_stage_final_reward_mean",
                "std": "reactive_stage_final_reward_std",
            }
        )
    )
    summary = summary.drop(
        columns=[
            "reactive_stage_125_reward",
            "reactive_stage_final_reward_mean",
            "reactive_stage_final_reward_std",
        ],
        errors="ignore",
    ).merge(
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


def with_seed(base_config: dict, *, seed: int) -> dict:
    config = copy.deepcopy(base_config)
    config["seed"] = int(seed)
    return config


def apply_reward_overrides(config: dict, variant_config: dict) -> dict:
    updated = copy.deepcopy(config)
    if "reward" in variant_config:
        updated.setdefault("reward", {}).update(variant_config["reward"])
    if "env" in variant_config:
        updated.setdefault("env", {}).update(variant_config["env"])
    return updated


def write_continuous_forgetting_plot(
    trajectory: pd.DataFrame,
    *,
    checkpoint_ids: list[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = trajectory.copy()
    if "stage" not in frame.columns:
        stage_by_checkpoint = {
            checkpoint: index + 1 for index, checkpoint in enumerate(checkpoint_ids)
        }
        frame["stage"] = frame["checkpoint"].map(stage_by_checkpoint)
    if "seed" not in frame.columns:
        frame["seed"] = 0
    if "naive_rolling_bwt" not in frame.columns:
        baseline = frame.groupby(["mapping", "seed"])["naive_task_1_reward"].transform("first")
        frame["naive_rolling_bwt"] = frame["naive_task_1_reward"] - baseline
        frame["replay_rolling_bwt"] = frame["replay_task_1_reward"] - baseline

    mappings = frame["mapping"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(mappings),
        1,
        figsize=(8.8, max(3.0, 2.7 * len(mappings))),
        sharex=True,
    )
    if len(mappings) == 1:
        axes = [axes]

    method_columns = [
        ("naive_rolling_bwt", "Naive Fine-Tuning", "tab:red"),
        ("replay_rolling_bwt", "PPO + Replay", "tab:blue"),
    ]
    for axis, mapping in zip(axes, mappings, strict=True):
        subset = frame[frame["mapping"] == mapping]
        for column, label, color in method_columns:
            grouped = (
                subset.groupby("stage", sort=True)[column]
                .agg(["mean", "std"])
                .reindex(range(1, len(checkpoint_ids) + 1))
            )
            x = grouped.index.to_numpy(dtype=float)
            mean = grouped["mean"].to_numpy(dtype=float)
            std = grouped["std"].fillna(0.0).to_numpy(dtype=float)
            axis.plot(x, mean, marker="o", linewidth=2.0, color=color, label=label)
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
        axis.axhline(0.0, color="0.45", linewidth=0.9)
        axis.axvline(1.5, color="0.25", linestyle="--", linewidth=1.0)
        axis.text(1.52, 0.96, "drift onset", transform=axis.get_xaxis_transform(), fontsize=8)
        axis.set_ylabel("Task 1 BWT")
        axis.set_title(f"{mapping.capitalize()} mapping")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, loc="best")

    axes[-1].set_xlabel("Training Stage")
    axes[-1].set_xticks(range(1, len(checkpoint_ids) + 1))
    axes[-1].set_xticklabels([str(checkpoint) for checkpoint in checkpoint_ids])
    figure.suptitle("Signed Backward Transfer Across Azure Checkpoints", y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def to_markdown_table(frame: pd.DataFrame) -> str:
    headers = [
        "Mapping",
        "Source",
        "VMs",
        "Seeds",
        "Initial Task 1",
        "Naive Final",
        "Replay Final",
        "Naive BWT",
        "Replay BWT",
        "Naive Forgetting",
        "Replay Forgetting",
    ]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.to_dict("records"):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["mapping"]),
                    str(row.get("curriculum_source", "azure")),
                    f"{int(row['selected_vm_count'])}",
                    f"{int(row['seed_count'])}",
                    mean_std(row, "initial_task_1_reward"),
                    mean_std(row, "naive_stage_final_reward"),
                    mean_std(row, "replay_stage_final_reward"),
                    mean_std(row, "naive_final_bwt"),
                    mean_std(row, "replay_final_bwt"),
                    mean_std(row, "naive_final_forgetting"),
                    mean_std(row, "replay_final_forgetting"),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def mean_std(row: dict[str, Any], prefix: str) -> str:
    mean = float(row[f"{prefix}_mean"])
    std = float(row[f"{prefix}_std"])
    if np.isnan(std):
        std = 0.0
    return f"{mean:.3f} +/- {std:.3f}"


if __name__ == "__main__":
    main()
