"""Replay mix ratio ablation on the linear mapping.

Sweeps ``replay_mix_ratio`` over a configurable set (default 0.0, 0.25, 0.5, 0.75) and
records the final-stage Task-1 BWT and mean-prior BWT per seed. The 0.0 setting should
recover Naive within noise — its plotted point is a sanity check on the mixing logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from calibrate_baselines import calibrate_static_p95
from run_drift_experiment import build_single_task_config, write_calibrated_trace
from run_sensitivity_analysis import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_SEED_RANGE_END,
    DEFAULT_SEED_RANGE_START,
    apply_reward_overrides,
    bridge_trace_boundaries,
    build_continuous_replay_config,
    evaluate_on_task,
    load_curriculum,
    preprocess_task,
    with_seed,
    with_training_overrides,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from driftscale.agents.train_ppo import train_maskable_ppo
from driftscale.agents.train_ppo_replay import train_replay_ppo
from driftscale.eval.forgetting import rolling_mean_prior_bwt
from driftscale.eval.stats import bootstrap_mean_ci
from driftscale.traces.azure_loader import AZURE_CHECKPOINT_IDS, VmSelectionStrategy
from driftscale.traces.preprocess import write_preprocessed_trace
from driftscale.utils.config import load_yaml
from driftscale.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-config", default="configs/env/linear.yaml")
    parser.add_argument("--ppo-config", default="configs/train/ppo.yaml")
    parser.add_argument("--replay-config", default="configs/train/ppo_replay.yaml")
    parser.add_argument("--output-dir", default="results/replay_ratio_ablation")
    parser.add_argument("--plot-path", default="media/replay_ratio_ablation.png")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--mix-ratios", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75]
    )
    parser.add_argument("--checkpoints", nargs="+", type=int, default=list(AZURE_CHECKPOINT_IDS))
    parser.add_argument("--vm-count", type=int, default=32)
    parser.add_argument(
        "--vm-selection-strategy",
        choices=[strategy.value for strategy in VmSelectionStrategy],
        default=VmSelectionStrategy.PER_CHECKPOINT_DENSE.value,
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--task-1-timesteps", type=int)
    parser.add_argument("--finetune-timesteps", type=int)
    parser.add_argument(
        "--curriculum-source",
        choices=("auto", "azure", "synthetic"),
        default="auto",
    )
    parser.add_argument("--min-drift-smd", type=float, default=0.5)
    parser.add_argument("--min-drift-ks", type=float, default=0.30)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ppo_base = load_yaml(args.ppo_config)
    replay_base = load_yaml(args.replay_config)

    seeds = resolve_seeds(args, ppo_base)
    curriculum = load_curriculum(args, seed=int(ppo_base.get("seed", DEFAULT_SEED_RANGE_START)))
    variant_config = load_yaml(args.mapping_config)
    variant = str(variant_config["mapping"]["variant"])
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_paths = [
        variant_dir / f"task_{checkpoint_id}_preprocessed.csv"
        for checkpoint_id in curriculum.checkpoint_ids
    ]
    traces = [
        preprocess_task(variant_config, matrix, path)
        for matrix, path in zip(curriculum.matrices, preprocessed_paths, strict=True)
    ]
    traces = bridge_trace_boundaries(traces)
    for trace, path in zip(traces, preprocessed_paths, strict=True):
        write_preprocessed_trace(trace, path)

    calibration_cfg = variant_config.get("calibration", {})
    calibration = calibrate_static_p95(
        traces[0]["demand"].to_numpy(dtype="float32"),
        min_tasks=int(ppo_base.get("env", {}).get("min_tasks", 1)),
        max_tasks=int(ppo_base.get("env", {}).get("max_tasks", 20)),
        target_min=float(calibration_cfg.get("target_slo_min", 0.01)),
        target_max=float(calibration_cfg.get("target_slo_max", 0.05)),
    )
    capacity_per_task = float(calibration["capacity_per_task"])
    calibrated_paths = [
        write_calibrated_trace(
            trace,
            variant_dir / f"task_{checkpoint_id}_calibrated.csv",
            scale_factor=float(calibration["scale_factor"]),
            capacity_per_task=capacity_per_task,
        )
        for checkpoint_id, trace in zip(curriculum.checkpoint_ids, traces, strict=True)
    ]

    rows: list[dict[str, float | int | str]] = []
    for mix_ratio in args.mix_ratios:
        for seed in seeds:
            row = run_single_ratio(
                mix_ratio=mix_ratio,
                seed=seed,
                variant=variant,
                variant_config=variant_config,
                variant_dir=variant_dir,
                checkpoint_ids=curriculum.checkpoint_ids,
                calibrated_paths=calibrated_paths,
                capacity_per_task=capacity_per_task,
                ppo_base=ppo_base,
                replay_base=replay_base,
                task_1_timesteps=args.task_1_timesteps,
                finetune_timesteps=args.finetune_timesteps,
            )
            rows.append(row)

    raw = pd.DataFrame(rows)
    raw_path = output_dir / "per_seed_bwt.csv"
    raw.to_csv(raw_path, index=False)

    summary = summarize_ablation(
        raw, bootstrap_resamples=args.bootstrap_resamples, alpha=args.alpha
    )
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    write_ablation_plot(summary, output_path=Path(args.plot_path))
    summary_markdown = summary_markdown_table(summary)
    (output_dir / "summary.md").write_text(summary_markdown + "\n", encoding="utf-8")
    print(summary_markdown)
    print(f"Saved per-seed BWT to {raw_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved plot to {args.plot_path}")


def resolve_seeds(args: argparse.Namespace, ppo_base: dict) -> list[int]:
    if args.seeds:
        return list(dict.fromkeys(args.seeds))
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    start_seed = int(ppo_base.get("seed", DEFAULT_SEED_RANGE_START))
    return [start_seed + offset for offset in range(args.seed_count)]


def run_single_ratio(
    *,
    mix_ratio: float,
    seed: int,
    variant: str,
    variant_config: dict,
    variant_dir: Path,
    checkpoint_ids: list[int],
    calibrated_paths: list[Path],
    capacity_per_task: float,
    ppo_base: dict,
    replay_base: dict,
    task_1_timesteps: int | None,
    finetune_timesteps: int | None,
) -> dict[str, float | int | str]:
    seed_everything(seed)
    training_cfg = dict(variant_config.get("training", {}))
    training_cfg["replay_mix_ratio"] = float(mix_ratio)
    if task_1_timesteps is not None:
        training_cfg["continuous_task_1_timesteps"] = int(task_1_timesteps)
    if finetune_timesteps is not None:
        training_cfg["continuous_finetune_timesteps"] = int(finetune_timesteps)

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
    replay_variant = with_seed(
        apply_reward_overrides(
            with_training_overrides(replay_base, finetune_training), variant_config
        ),
        seed=seed,
    )

    n_stages = len(checkpoint_ids)
    matrix = np.full((n_stages, n_stages), np.nan, dtype=np.float64)

    seed_dir = variant_dir / f"seed_{seed}_mix_{mix_ratio:.2f}"
    task_1_config = build_single_task_config(
        ppo_task_1,
        trace_path=calibrated_paths[0],
        output_dir=seed_dir / "stage_1",
        capacity_per_task=capacity_per_task,
    )
    task_1_metrics = train_maskable_ppo(task_1_config)
    matrix[0, 0] = evaluate_on_task(
        task_1_config,
        task_path=calibrated_paths[0],
        model_path=task_1_metrics["model_path"],
        vecnormalize_path=task_1_metrics["vecnormalize_path"],
    )

    current_metrics = task_1_metrics
    for stage_index, checkpoint_id in enumerate(checkpoint_ids[1:], start=1):
        replay_config = build_continuous_replay_config(
            replay_variant,
            current_task_path=calibrated_paths[stage_index],
            previous_task_paths=calibrated_paths[:stage_index],
            output_dir=seed_dir / f"stage_{checkpoint_id}",
            capacity_per_task=capacity_per_task,
            init_model_path=current_metrics["model_path"],
            init_vecnormalize_path=current_metrics["vecnormalize_path"],
            replay_mix_ratio=float(mix_ratio),
            n_envs=int(training_cfg.get("n_envs", 4)),
        )
        current_metrics = train_replay_ppo(replay_config)
        for prior_index in range(stage_index + 1):
            matrix[prior_index, stage_index] = evaluate_on_task(
                replay_config,
                task_path=calibrated_paths[prior_index],
                model_path=current_metrics["model_path"],
                vecnormalize_path=current_metrics["vecnormalize_path"],
            )

    final_stage = n_stages - 1
    task_1_bwt = matrix[0, final_stage] - matrix[0, 0]
    mean_prior_bwt = rolling_mean_prior_bwt(matrix)[final_stage]
    return {
        "mapping": variant,
        "mix_ratio": float(mix_ratio),
        "seed": int(seed),
        "final_task_1_bwt": float(task_1_bwt),
        "final_mean_prior_bwt": float(mean_prior_bwt),
    }


def summarize_ablation(
    raw: pd.DataFrame, *, bootstrap_resamples: int, alpha: float
) -> pd.DataFrame:
    grouped = raw.groupby("mix_ratio", sort=True)
    summary_rows = []
    for mix_ratio, group in grouped:
        task1 = group["final_task_1_bwt"].to_numpy(dtype=np.float64)
        meanprior = group["final_mean_prior_bwt"].to_numpy(dtype=np.float64)
        bootstrap_seed = int(round(mix_ratio * 1000))
        task1_ci = bootstrap_mean_ci(
            task1, resamples=bootstrap_resamples, alpha=alpha, seed=bootstrap_seed
        )
        prior_ci = bootstrap_mean_ci(
            meanprior, resamples=bootstrap_resamples, alpha=alpha, seed=bootstrap_seed + 1
        )
        summary_rows.append(
            {
                "mix_ratio": float(mix_ratio),
                "seed_count": len(group),
                "task_1_bwt_mean": task1_ci.mean,
                "task_1_bwt_ci_low": task1_ci.ci_low,
                "task_1_bwt_ci_high": task1_ci.ci_high,
                "mean_prior_bwt_mean": prior_ci.mean,
                "mean_prior_bwt_ci_low": prior_ci.ci_low,
                "mean_prior_bwt_ci_high": prior_ci.ci_high,
            }
        )
    return pd.DataFrame(summary_rows)


def write_ablation_plot(summary: pd.DataFrame, *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    ratios = summary["mix_ratio"].to_numpy(dtype=float)

    for column_prefix, label, color in [
        ("task_1_bwt", "Task-1 BWT", "tab:red"),
        ("mean_prior_bwt", "Mean prior BWT", "tab:blue"),
    ]:
        means = summary[f"{column_prefix}_mean"].to_numpy(dtype=float)
        lows = summary[f"{column_prefix}_ci_low"].to_numpy(dtype=float)
        highs = summary[f"{column_prefix}_ci_high"].to_numpy(dtype=float)
        lower_err = means - lows
        upper_err = highs - means
        axis.errorbar(
            ratios,
            means,
            yerr=[lower_err, upper_err],
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
            capsize=4,
        )
    axis.axhline(0.0, color="0.45", linewidth=0.9)
    axis.set_xlabel("replay_mix_ratio (fraction of envs replaying prior tasks)")
    axis.set_ylabel("Final-stage BWT")
    axis.set_title("Replay-mix-ratio ablation (linear mapping, bootstrap 95% CI)")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def summary_markdown_table(summary: pd.DataFrame) -> str:
    rows = [
        "| Mix Ratio | n | Task-1 BWT (95% CI) | Mean-prior BWT (95% CI) |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary.to_dict("records"):
        rows.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['mix_ratio']):.2f}",
                    f"{int(row['seed_count'])}",
                    (
                        f"{float(row['task_1_bwt_mean']):+.2f} "
                        f"[{float(row['task_1_bwt_ci_low']):+.2f}, "
                        f"{float(row['task_1_bwt_ci_high']):+.2f}]"
                    ),
                    (
                        f"{float(row['mean_prior_bwt_mean']):+.2f} "
                        f"[{float(row['mean_prior_bwt_ci_low']):+.2f}, "
                        f"{float(row['mean_prior_bwt_ci_high']):+.2f}]"
                    ),
                ]
            )
            + " |"
        )
    # Used by callers to keep the variable referenced; guards against unused-import false positives.
    _ = (DEFAULT_BOOTSTRAP_RESAMPLES, DEFAULT_ALPHA, DEFAULT_SEED_RANGE_END)
    return "\n".join(rows)


if __name__ == "__main__":
    main()
