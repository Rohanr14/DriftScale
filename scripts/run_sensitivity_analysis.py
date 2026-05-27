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
from driftscale.eval.forgetting import rolling_mean_prior_bwt
from driftscale.eval.stats import bootstrap_mean_ci, paired_wilcoxon
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
from driftscale.utils.seeding import seed_everything

METHOD_COLORS = {
    "naive": "tab:red",
    "replay": "tab:blue",
    "reactive": "tab:gray",
}
METHOD_LABELS = {
    "naive": "Naive Fine-Tuning",
    "replay": "PPO + Replay",
    "reactive": "Reactive (sanity)",
}
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED_RANGE_START = 7
DEFAULT_SEED_RANGE_END = 12  # seeds 7..11 inclusive — extends original {7,8,9}


@dataclass(frozen=True)
class ContinuousVariantResult:
    """Summary, per-seed trajectory, and audit tables for one §5.5 mapping variant."""

    summary: dict[str, float | int | str]
    trajectory: pd.DataFrame
    diagnostics: pd.DataFrame
    per_seed_bwt: pd.DataFrame
    per_stage_rewards: pd.DataFrame


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
        default=DEFAULT_SEED_RANGE_END - DEFAULT_SEED_RANGE_START,
        help=(
            "Number of consecutive seeds to run, starting at the PPO config seed unless "
            "--seeds is supplied. Default 5 (seeds 7..11). Runtime scales roughly linearly."
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
            "to labeled synthetic regimes otherwise."
        ),
    )
    parser.add_argument("--min-drift-smd", type=float, default=0.5)
    parser.add_argument("--min-drift-ks", type=float, default=0.30)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="Percentile bootstrap resample count for BWT mean CIs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Statistical alpha for bootstrap CIs and significance thresholds.",
    )
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
        diagnostics_path = output_dir / "demand_diagnostics.csv"
        diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else None
        write_continuous_forgetting_plot(
            trajectories,
            checkpoint_ids=list(args.checkpoints),
            output_path=Path(args.plot_path),
            diagnostics=diagnostics,
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
            bootstrap_resamples=args.bootstrap_resamples,
            alpha=args.alpha,
        )
        for config_path in mapping_configs
    ]

    summary = pd.DataFrame([result.summary for result in results])
    trajectories = pd.concat([result.trajectory for result in results], ignore_index=True)
    diagnostics = pd.concat(
        [validation_diagnostics, *[result.diagnostics for result in results]],
        ignore_index=True,
    )
    per_seed_bwt = pd.concat([result.per_seed_bwt for result in results], ignore_index=True)
    per_stage_rewards = pd.concat(
        [result.per_stage_rewards for result in results], ignore_index=True
    )

    summary_path = output_dir / "summary.csv"
    markdown_path = output_dir / "summary.md"
    trajectory_path = output_dir / "continuous_rewards.csv"
    diagnostics_path = output_dir / "demand_diagnostics.csv"
    per_seed_path = output_dir / "per_seed_bwt.csv"
    per_stage_rewards_path = output_dir / "per_stage_rewards.csv"

    summary.to_csv(summary_path, index=False)
    trajectories.to_csv(trajectory_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    per_seed_bwt.to_csv(per_seed_path, index=False)
    per_stage_rewards.to_csv(per_stage_rewards_path, index=False)

    write_continuous_forgetting_plot(
        trajectories,
        checkpoint_ids=curriculum.checkpoint_ids,
        output_path=Path(args.plot_path),
        diagnostics=diagnostics,
    )

    markdown = compose_summary_markdown(summary, diagnostics=diagnostics)
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


def resolve_seeds(args: argparse.Namespace, ppo_base: dict[str, Any]) -> list[int]:
    if args.seeds:
        return list(dict.fromkeys(args.seeds))
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    start_seed = int(ppo_base.get("seed", DEFAULT_SEED_RANGE_START))
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
    bootstrap_resamples: int,
    alpha: float,
) -> ContinuousVariantResult:
    variant_config = load_yaml(config_path)
    variant = str(variant_config["mapping"]["variant"])
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_paths = [
        variant_dir / f"task_{checkpoint_id}_preprocessed.csv" for checkpoint_id in checkpoint_ids
    ]
    traces = [
        preprocess_task(variant_config, matrix, output_path)
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
    reactive_rewards_per_stage = evaluate_reactive_stage_rewards(
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

    seed_results = [
        run_mapping_seed(
            variant=variant,
            variant_config=variant_config,
            variant_dir=variant_dir,
            checkpoint_ids=checkpoint_ids,
            calibrated_paths=calibrated_paths,
            reactive_rewards_per_stage=reactive_rewards_per_stage,
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
    trajectory = pd.concat([sr["trajectory"] for sr in seed_results], ignore_index=True)
    per_seed_bwt = pd.concat([sr["per_seed"] for sr in seed_results], ignore_index=True)
    per_stage_rewards = pd.concat(
        [sr["per_stage_rewards"] for sr in seed_results], ignore_index=True
    )
    trajectory.to_csv(variant_dir / "continuous_rewards.csv", index=False)
    per_seed_bwt.to_csv(variant_dir / "per_seed_bwt.csv", index=False)
    per_stage_rewards.to_csv(variant_dir / "per_stage_rewards.csv", index=False)

    return ContinuousVariantResult(
        summary=summarize_variant(
            trajectory,
            per_seed_bwt=per_seed_bwt,
            mapping=variant,
            selected_vm_count=selected_vm_count,
            vm_selection_strategy=vm_selection_strategy,
            curriculum_source=curriculum_source,
            seeds=seeds,
            bootstrap_resamples=bootstrap_resamples,
            alpha=alpha,
        ),
        trajectory=trajectory,
        diagnostics=diagnostics,
        per_seed_bwt=per_seed_bwt,
        per_stage_rewards=per_stage_rewards,
    )


def run_mapping_seed(
    *,
    variant: str,
    variant_config: dict[str, Any],
    variant_dir: Path,
    checkpoint_ids: list[int],
    calibrated_paths: list[Path],
    reactive_rewards_per_stage: list[float],
    capacity_per_task: float,
    ppo_base: dict[str, Any],
    replay_base: dict[str, Any],
    seed: int,
    task_1_timesteps: int | None,
    finetune_timesteps: int | None,
    replay_mix_ratio: float | None,
) -> dict[str, pd.DataFrame]:
    seed_everything(seed)
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

    n_stages = len(checkpoint_ids)
    naive_matrix = np.full((n_stages, n_stages), np.nan, dtype=np.float64)
    replay_matrix = np.full((n_stages, n_stages), np.nan, dtype=np.float64)

    seed_dir = variant_dir / f"seed_{seed}"
    task_1_config = build_single_task_config(
        ppo_task_1,
        trace_path=calibrated_paths[0],
        output_dir=seed_dir / "naive_stage_1",
        capacity_per_task=capacity_per_task,
    )
    task_1_metrics = train_maskable_ppo(task_1_config)
    naive_matrix[0, 0] = evaluate_on_task(
        task_1_config,
        task_path=calibrated_paths[0],
        model_path=task_1_metrics["model_path"],
        vecnormalize_path=task_1_metrics["vecnormalize_path"],
    )
    replay_matrix[0, 0] = naive_matrix[0, 0]  # identical model at stage 0

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
        for prior_index in range(stage_index + 1):
            naive_matrix[prior_index, stage_index] = evaluate_on_task(
                naive_config,
                task_path=calibrated_paths[prior_index],
                model_path=naive_metrics["model_path"],
                vecnormalize_path=naive_metrics["vecnormalize_path"],
            )

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
        for prior_index in range(stage_index + 1):
            replay_matrix[prior_index, stage_index] = evaluate_on_task(
                replay_config,
                task_path=calibrated_paths[prior_index],
                model_path=replay_metrics["model_path"],
                vecnormalize_path=replay_metrics["vecnormalize_path"],
            )

    reactive_matrix = build_reactive_matrix(
        calibrated_paths,
        base_config=ppo_base,
        variant_config=variant_config,
        capacity_per_task=capacity_per_task,
        n_stages=n_stages,
    )

    naive_task_1_rewards = naive_matrix[0, :].tolist()
    replay_task_1_rewards = replay_matrix[0, :].tolist()
    reactive_task_1_rewards = reactive_matrix[0, :].tolist()

    initial_reward = naive_task_1_rewards[0]
    naive_task_1_bwt = np.asarray(naive_task_1_rewards, dtype=np.float64) - initial_reward
    replay_task_1_bwt = np.asarray(replay_task_1_rewards, dtype=np.float64) - initial_reward
    reactive_task_1_bwt = (
        np.asarray(reactive_task_1_rewards, dtype=np.float64) - reactive_task_1_rewards[0]
    )

    naive_mean_prior = rolling_mean_prior_bwt(naive_matrix)
    replay_mean_prior = rolling_mean_prior_bwt(replay_matrix)
    reactive_mean_prior = rolling_mean_prior_bwt(reactive_matrix)

    trajectory = pd.DataFrame(
        {
            "mapping": variant,
            "seed": seed,
            "checkpoint": checkpoint_ids,
            "stage": list(range(1, n_stages + 1)),
            "naive_task_1_reward": naive_task_1_rewards,
            "replay_task_1_reward": replay_task_1_rewards,
            "reactive_stage_reward": reactive_rewards_per_stage,
            "reactive_task_1_reward": reactive_task_1_rewards,
            "naive_rolling_bwt": naive_task_1_bwt,
            "replay_rolling_bwt": replay_task_1_bwt,
            "reactive_rolling_bwt": reactive_task_1_bwt,
            "naive_mean_prior_bwt": naive_mean_prior,
            "replay_mean_prior_bwt": replay_mean_prior,
            "reactive_mean_prior_bwt": reactive_mean_prior,
            "naive_forgetting": np.maximum(0.0, -naive_task_1_bwt),
            "replay_forgetting": np.maximum(0.0, -replay_task_1_bwt),
        }
    )

    final_stage = n_stages - 1
    per_seed = pd.DataFrame(
        [
            {
                "mapping": variant,
                "seed": seed,
                "method": "naive",
                "final_task_1_bwt": float(naive_task_1_bwt[final_stage]),
                "final_mean_prior_bwt": float(naive_mean_prior[final_stage]),
            },
            {
                "mapping": variant,
                "seed": seed,
                "method": "replay",
                "final_task_1_bwt": float(replay_task_1_bwt[final_stage]),
                "final_mean_prior_bwt": float(replay_mean_prior[final_stage]),
            },
            {
                "mapping": variant,
                "seed": seed,
                "method": "reactive",
                "final_task_1_bwt": float(reactive_task_1_bwt[final_stage]),
                "final_mean_prior_bwt": float(reactive_mean_prior[final_stage]),
            },
        ]
    )

    per_stage_rewards = matrices_to_long_form(
        mapping=variant,
        seed=seed,
        checkpoint_ids=checkpoint_ids,
        matrices={
            "naive": naive_matrix,
            "replay": replay_matrix,
            "reactive": reactive_matrix,
        },
    )

    trajectory.to_csv(seed_dir / "continuous_rewards.csv", index=False)
    return {
        "trajectory": trajectory,
        "per_seed": per_seed,
        "per_stage_rewards": per_stage_rewards,
    }


def evaluate_on_task(
    config: dict[str, Any],
    *,
    task_path: Path,
    model_path: str | Path,
    vecnormalize_path: str | Path,
) -> float:
    """Return the total reward of a saved policy on one task trace."""
    metrics = evaluate_task_a(
        config,
        task_a_path=task_path,
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
    )
    return float(metrics["total_reward"])


def build_reactive_matrix(
    calibrated_paths: list[Path],
    *,
    base_config: dict[str, Any],
    variant_config: dict[str, Any],
    capacity_per_task: float,
    n_stages: int,
) -> np.ndarray:
    """Build the reactive reward matrix. By construction every column duplicates column 0."""
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
    matrix = np.full((n_stages, n_stages), np.nan, dtype=np.float64)
    rewards: list[float] = []
    for trace_path in calibrated_paths:
        demand = pd.read_csv(trace_path)["demand"].to_numpy(dtype=np.float32)
        rewards.append(float(autoscaler.evaluate(demand).total_reward))
    for column in range(n_stages):
        for row in range(column + 1):
            matrix[row, column] = rewards[row]
    return matrix


def matrices_to_long_form(
    *,
    mapping: str,
    seed: int,
    checkpoint_ids: list[int],
    matrices: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, matrix in matrices.items():
        n_stages = matrix.shape[0]
        for stage_col in range(n_stages):
            for task_row in range(n_stages):
                reward = matrix[task_row, stage_col]
                if np.isnan(reward):
                    continue
                rows.append(
                    {
                        "mapping": mapping,
                        "seed": seed,
                        "method": method,
                        "train_stage": stage_col + 1,
                        "train_checkpoint": checkpoint_ids[stage_col],
                        "eval_task_stage": task_row + 1,
                        "eval_task_checkpoint": checkpoint_ids[task_row],
                        "reward": float(reward),
                    }
                )
    return pd.DataFrame(rows)


def summarize_variant(
    trajectory: pd.DataFrame,
    *,
    per_seed_bwt: pd.DataFrame,
    mapping: str,
    selected_vm_count: int,
    vm_selection_strategy: str,
    curriculum_source: str,
    seeds: list[int],
    bootstrap_resamples: int,
    alpha: float,
) -> dict[str, float | int | str]:
    final_stage = int(trajectory["stage"].max())
    final_rows = trajectory[trajectory["stage"] == final_stage].sort_values("seed")
    summary: dict[str, float | int | str] = {
        "mapping": mapping,
        "curriculum_source": curriculum_source,
        "vm_selection_strategy": vm_selection_strategy,
        "selected_vm_count": selected_vm_count,
        "seed_count": len(seeds),
        "seeds": ",".join(str(seed) for seed in seeds),
        "final_checkpoint": int(final_rows["checkpoint"].iloc[0]),
    }
    initial_rewards = (
        trajectory[trajectory["stage"] == 1].sort_values("seed")["naive_task_1_reward"].to_numpy()
    )
    summary["initial_task_1_reward_mean"] = float(np.mean(initial_rewards))
    summary["initial_task_1_reward_std"] = _safe_std(initial_rewards)

    summary["naive_stage_final_reward_mean"] = float(np.mean(final_rows["naive_task_1_reward"]))
    summary["naive_stage_final_reward_std"] = _safe_std(
        final_rows["naive_task_1_reward"].to_numpy()
    )
    summary["replay_stage_final_reward_mean"] = float(np.mean(final_rows["replay_task_1_reward"]))
    summary["replay_stage_final_reward_std"] = _safe_std(
        final_rows["replay_task_1_reward"].to_numpy()
    )
    summary["reactive_stage_final_reward_mean"] = float(
        np.mean(final_rows["reactive_stage_reward"])
    )
    summary["reactive_stage_final_reward_std"] = _safe_std(
        final_rows["reactive_stage_reward"].to_numpy()
    )

    final_by_method = per_seed_bwt[per_seed_bwt["mapping"] == mapping]
    naive_seed_bwt = final_by_method[final_by_method["method"] == "naive"].sort_values("seed")
    replay_seed_bwt = final_by_method[final_by_method["method"] == "replay"].sort_values("seed")
    reactive_seed_bwt = final_by_method[final_by_method["method"] == "reactive"].sort_values(
        "seed"
    )

    naive_task1 = naive_seed_bwt["final_task_1_bwt"].to_numpy()
    replay_task1 = replay_seed_bwt["final_task_1_bwt"].to_numpy()
    reactive_task1 = reactive_seed_bwt["final_task_1_bwt"].to_numpy()
    naive_meanprior = naive_seed_bwt["final_mean_prior_bwt"].to_numpy()
    replay_meanprior = replay_seed_bwt["final_mean_prior_bwt"].to_numpy()
    reactive_meanprior = reactive_seed_bwt["final_mean_prior_bwt"].to_numpy()

    bootstrap_seed = sum(seeds) % (2**32)
    for method_name, values in [
        ("naive_final_bwt", naive_task1),
        ("replay_final_bwt", replay_task1),
        ("reactive_final_bwt", reactive_task1),
        ("naive_final_mean_prior_bwt", naive_meanprior),
        ("replay_final_mean_prior_bwt", replay_meanprior),
        ("reactive_final_mean_prior_bwt", reactive_meanprior),
    ]:
        clean = values[~np.isnan(values)]
        ci = bootstrap_mean_ci(
            clean,
            resamples=bootstrap_resamples,
            alpha=alpha,
            seed=bootstrap_seed,
        )
        summary[f"{method_name}_mean"] = ci.mean
        summary[f"{method_name}_std"] = _safe_std(clean)
        summary[f"{method_name}_ci_low"] = ci.ci_low
        summary[f"{method_name}_ci_high"] = ci.ci_high
        summary[f"{method_name}_n"] = ci.n

    task1_p = paired_wilcoxon(naive_task1, replay_task1)
    summary["naive_vs_replay_task1_wilcoxon_p"] = task1_p.p_value
    summary["naive_vs_replay_task1_median_diff"] = task1_p.median_difference

    valid_prior = ~(np.isnan(naive_meanprior) | np.isnan(replay_meanprior))
    if valid_prior.any():
        meanprior_p = paired_wilcoxon(naive_meanprior[valid_prior], replay_meanprior[valid_prior])
        summary["naive_vs_replay_mean_prior_wilcoxon_p"] = meanprior_p.p_value
        summary["naive_vs_replay_mean_prior_median_diff"] = meanprior_p.median_difference
    else:
        summary["naive_vs_replay_mean_prior_wilcoxon_p"] = float("nan")
        summary["naive_vs_replay_mean_prior_median_diff"] = float("nan")

    summary["naive_final_forgetting_mean"] = float(np.mean(np.maximum(0.0, -naive_task1)))
    summary["naive_final_forgetting_std"] = _safe_std(np.maximum(0.0, -naive_task1))
    summary["replay_final_forgetting_mean"] = float(np.mean(np.maximum(0.0, -replay_task1)))
    summary["replay_final_forgetting_std"] = _safe_std(np.maximum(0.0, -replay_task1))
    summary["bootstrap_resamples"] = bootstrap_resamples
    summary["alpha"] = alpha
    return summary


def _safe_std(values: np.ndarray | list[float]) -> float:
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size < 2:
        return 0.0
    std = float(np.std(array, ddof=1))
    return 0.0 if np.isnan(std) else std


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
    diagnostics_path = output_dir / "demand_diagnostics.csv"
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else None
    markdown_path.write_text(
        compose_summary_markdown(summary, diagnostics=diagnostics) + "\n",
        encoding="utf-8",
    )


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
    diagnostics: pd.DataFrame | None = None,
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

    has_mean_prior = (
        "naive_mean_prior_bwt" in frame.columns and "replay_mean_prior_bwt" in frame.columns
    )

    mappings = frame["mapping"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(mappings),
        1,
        figsize=(9.2, max(3.4, 3.0 * len(mappings))),
        sharex=True,
    )
    if len(mappings) == 1:
        axes = [axes]

    primary_columns = (
        [("naive_mean_prior_bwt", "naive", "Naive (mean prior BWT)"),
         ("replay_mean_prior_bwt", "replay", "Replay (mean prior BWT)")]
        if has_mean_prior
        else [("naive_rolling_bwt", "naive", "Naive (Task-1 BWT)"),
              ("replay_rolling_bwt", "replay", "Replay (Task-1 BWT)")]
    )

    for axis, mapping in zip(axes, mappings, strict=True):
        subset = frame[frame["mapping"] == mapping]
        for column, method_key, label in primary_columns:
            color = METHOD_COLORS[method_key]
            grouped = subset.groupby("stage", sort=True)[column]
            stages = sorted(subset["stage"].unique())
            means = []
            ci_lo: list[float] = []
            ci_hi: list[float] = []
            for stage in stages:
                values = (
                    grouped.get_group(stage).to_numpy(dtype=np.float64)
                    if stage in grouped.groups
                    else np.array([])
                )
                clean = values[~np.isnan(values)]
                if clean.size == 0:
                    means.append(np.nan)
                    ci_lo.append(np.nan)
                    ci_hi.append(np.nan)
                    continue
                ci = bootstrap_mean_ci(clean, resamples=2000, seed=int(sum(stages) + stage))
                means.append(ci.mean)
                ci_lo.append(ci.ci_low)
                ci_hi.append(ci.ci_high)
            xs = np.asarray(stages, dtype=float)
            axis.plot(xs, means, marker="o", linewidth=2.0, color=color, label=label)
            axis.fill_between(xs, ci_lo, ci_hi, color=color, alpha=0.18, linewidth=0)
        axis.axhline(0.0, color="0.45", linewidth=0.9)
        axis.axvline(1.5, color="0.25", linestyle="--", linewidth=1.0)
        axis.text(1.52, 0.96, "drift onset", transform=axis.get_xaxis_transform(), fontsize=8)
        axis.set_ylabel("BWT (mean prior)" if has_mean_prior else "Task-1 BWT")
        axis.set_title(f"{mapping.capitalize()} mapping")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, loc="best")

        if diagnostics is not None:
            demand_axis = axis.twiny()
            demand_axis.set_xlim(axis.get_xlim())
            mapping_diag = diagnostics[
                (diagnostics["mapping"] == mapping)
                & (diagnostics["curriculum_source"] != "synthetic_check")
            ].sort_values("stage")
            if not mapping_diag.empty:
                tick_stages = mapping_diag["stage"].to_numpy(dtype=float)
                demand_means = mapping_diag["mean"].to_numpy(dtype=float)
                demand_axis.set_xticks(tick_stages)
                demand_axis.set_xticklabels(
                    [f"{value:.2f}" for value in demand_means], fontsize=7, color="0.4"
                )
                demand_axis.set_xlabel("Per-checkpoint demand mean", fontsize=8, color="0.4")
            demand_axis.tick_params(axis="x", length=2)

    axes[-1].set_xlabel("Training Stage (checkpoint id)")
    axes[-1].set_xticks(range(1, len(checkpoint_ids) + 1))
    axes[-1].set_xticklabels(
        [f"{stage}\n(ckpt {checkpoint})" for stage, checkpoint in enumerate(checkpoint_ids, 1)]
    )
    figure.suptitle(
        "Backward Transfer Across Azure Checkpoints (bootstrap 95% CI)", y=0.995
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def compose_summary_markdown(
    summary: pd.DataFrame,
    *,
    diagnostics: pd.DataFrame | None = None,
) -> str:
    sections = [
        "## Final-stage BWT (signed; positive = no forgetting)",
        "",
        format_bwt_markdown(summary),
    ]
    if diagnostics is not None:
        sections.extend(
            [
                "",
                "## Per-checkpoint drift magnitude",
                "",
                format_drift_markdown(diagnostics),
            ]
        )
    return "\n".join(sections)


def format_bwt_markdown(summary: pd.DataFrame) -> str:
    headers = [
        "Mapping",
        "Source",
        "VMs",
        "Seeds",
        "Naive BWT (95% CI)",
        "Replay BWT (95% CI)",
        "Reactive BWT (sanity)",
        "Wilcoxon p (naive vs replay)",
        "Naive mean-prior BWT",
        "Replay mean-prior BWT",
    ]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in summary.to_dict("records"):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row.get("mapping", "")),
                    str(row.get("curriculum_source", "azure")),
                    f"{int(row.get('selected_vm_count', 0))}",
                    f"{int(row.get('seed_count', 0))}",
                    fmt_ci(row, "naive_final_bwt"),
                    fmt_ci(row, "replay_final_bwt"),
                    fmt_ci(row, "reactive_final_bwt"),
                    fmt_p(row.get("naive_vs_replay_task1_wilcoxon_p")),
                    fmt_ci(row, "naive_final_mean_prior_bwt"),
                    fmt_ci(row, "replay_final_mean_prior_bwt"),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def format_drift_markdown(diagnostics: pd.DataFrame) -> str:
    if diagnostics is None or diagnostics.empty:
        return "_no diagnostics_"
    # Use the curriculum-level diagnostic (linear_raw aggregate) for the headline table.
    aggregate = diagnostics[diagnostics["mapping"] == "linear_raw"]
    if aggregate.empty:
        aggregate = diagnostics
    aggregate = aggregate.drop_duplicates(subset=["checkpoint"]).sort_values("stage")
    headers = [
        "Checkpoint",
        "Demand mean",
        "Demand std",
        "SMD vs ckpt 1",
        "KS vs ckpt 1",
        "Different?",
    ]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in aggregate.to_dict("records"):
        smd = row.get("smd_vs_checkpoint_1")
        ks = row.get("ks_vs_checkpoint_1")
        smd_str = "—" if _is_missing(smd) else f"{float(smd):+.3f}"
        ks_str = "—" if _is_missing(ks) else f"{float(ks):.3f}"
        rows.append(
            "| "
            + " | ".join(
                [
                    str(int(row["checkpoint"])),
                    f"{float(row['mean']):.3f}",
                    f"{float(row['std']):.3f}",
                    smd_str,
                    ks_str,
                    "yes" if bool(row.get("measurable_shift", False)) else "—",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def fmt_ci(row: dict[str, Any], prefix: str) -> str:
    mean = row.get(f"{prefix}_mean")
    ci_low = row.get(f"{prefix}_ci_low")
    ci_high = row.get(f"{prefix}_ci_high")
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "n/a"
    if ci_low is None or ci_high is None:
        return f"{float(mean):.2f}"
    return f"{float(mean):+.2f} [{float(ci_low):+.2f}, {float(ci_high):+.2f}]"


def fmt_p(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if value < 0.001:
        return f"{float(value):.1e}"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
