# DriftScale

**DriftScale benchmarks how reinforcement learning (RL) autoscaling policies degrade and recover under workload drift, using public Azure VM traces with reactive autoscaling as a reference baseline and an AWS ECS Fargate control-loop validation.**

## Demo

Phase 1 through Week 4 are implemented and run locally: synthetic workload generation,
the `DriftScaleEnv` Gymnasium environment, reward accounting, action masks, static baselines,
reactive threshold autoscaling, Azure-shaped trace loading, trace-to-demand preprocessing,
baseline calibration, MaskablePPO training, and the first cost/SLO plot.

The later Loom/AWS demo will come after the Azure/PPO/replay phases.

## Current Status

**Phase 1 complete:** repo scaffold, synthetic env, reward model, reactive baseline, static
baselines, masked random-policy smoke run, and unit tests.

**Week 2 complete:** Azure CSV loader, timestamp alignment, deterministic VM subsetting,
vectorized §5.5 demand mappings, tiny local Azure-shaped sample, and cache generation.

**Weeks 3-4 complete:** baseline calibration into the static-p95 1-5% SLO band,
`MaskablePPO` with `VecNormalize`, saved PPO artifacts, and `media/cost_vs_slo.png`.

**Weeks 5-6 complete:** abrupt Task A -> Task B drift setup, BWT/forgetting metrics,
naive fine-tuning, and interleaved replay via mixed vectorized environments.

**Sensitivity suite complete:** real local Azure V1 checkpoint loading, per-checkpoint dense
32-VM cohort selection, drift diagnostics, multi-seed continuous fine-tuning/replay, and
§5.5 linear/convex/threshold BWT comparison with bootstrap CIs and Wilcoxon paired tests.

### Approximate runtimes (8-core CPU, no GPU)

| Target | Wall-clock | Notes |
| --- | --- | --- |
| `make test` | 5–10 s | unit suite only |
| `make lint` | 1–3 s | ruff |
| `make quickcheck` | 2–4 min | 2 seeds × tiny PPO budgets; pipeline-only smoke test |
| `make sensitivity-suite` | 25–50 min | 5 seeds × 3 mappings × 6 stages × naive+replay |
| `make replay-ratio-ablation` | 10–20 min | 4 mix ratios × 3 seeds on linear mapping |
| `make post-eval` | <1 min | re-plots from cached CSVs and existing model artifacts |

Validated locally:

```bash
make setup
make test
make lint
make preprocess
make calibrate
make train-baseline
make drift-experiment
make fetch-data
make quickcheck         # ~2 min pipeline smoke test (2 seeds, tiny PPO budgets)
make sensitivity-suite  # ~25–50 min: real numbers
make replay-ratio-ablation
make post-eval
make random-run
make phase1
```

## Why This Matters

"An RL autoscaler that beats reactive scaling" is a brittle claim. Reactive autoscaling is a highly robust, aggressively optimized baseline tuned over a decade of production experience. Instead of claiming a production-ready replacement, DriftScale aims at a different, highly relevant problem: **catastrophic forgetting in dynamic cloud environments.** When cloud workloads drift, learned policies degrade. DriftScale provides a reproducible empirical benchmark to quantify this degradation, measure how fast policies adapt to new regimes, and evaluate the continual-learning method (interleaved experience replay) used to mitigate forgetting.

## What DriftScale Does

This project is an end-to-end ML-for-systems benchmark that:

1. **Quantifies Catastrophic Forgetting:** Measures how vanilla PPO models forget past workload patterns when confronted with new ones (abrupt and gradual drift).
2. **Evaluates Continual Learning:** Tests replay-based interleaving against naive fine-tuning to measure how much forgetting it actually prevents on the same checkpoints. (EWC is scoped out for this iteration — see Limitations / Future Work.)
3. **Validates with Real Infrastructure:** Bridges the gap between simulation and reality by deploying the learned control loop against a live AWS ECS Fargate service, complete with safety guardrails and metric ingestion.

## Architecture

DriftScale operates on two distinct paths:

* **Simulation & Training Path:** Azure Public Traces → Preprocessor (Trace-to-Demand mapping) → Custom `DriftScaleEnv` (Gymnasium) → Policy Training (Reactive, PPO, PPO + Replay) → Evaluation.
* **Validation Path:** `k6` Trace Replay → AWS Application Load Balancer → ECS Fargate FastAPI Service → CloudWatch Metrics → Safety-Wrapped RL Controller → ECS `desiredCount` updates.

## Quickstart

This project is built with Python 3.11+ and uses `uv` for dependency management.

```bash
# 1. Install dependencies
make setup

# 2. Run test suite
make test

# 3. Build the tiny Azure V1-style demand cache
make preprocess

# 4. Calibrate scale/capacity and baseline metrics
make calibrate

# 5. Train the vanilla MaskablePPO baseline and generate the cost/SLO plot
make train-baseline

# 6. Run abrupt drift and replay forgetting evaluation
make drift-experiment

# 7. Materialize the top-1,000 dense VM matrix from the real local Azure shard
make fetch-data

# 8. Run six-checkpoint sensitivity across linear, convex, and threshold mappings
make sensitivity-suite

# 9. Rebuild the forgetting and rollout plots from cached CSV artifacts
make post-eval

# 10. Run the masked random-policy smoke test
make random-run

# 11. Evaluate Phase 1 static and reactive baselines
make phase1
```

## Results

The sensitivity suite reads the six real local Azure V1 checkpoint shards at
`data/raw/vm_cpu_readings-file-{1,25,50,75,100,125}-of-125.csv.gz`, selects each
checkpoint's own 32 densest VMs, and rejects the run unless later checkpoints measurably
differ from checkpoint 1 (`SMD ≥ 0.5` or `KS ≥ 0.30`). Outputs:

- `results/sensitivity/summary.{csv,md}` — final-stage BWT mean ± 95% CI + Wilcoxon p.
- `results/sensitivity/per_seed_bwt.csv` — raw per-seed Task-1 and mean-prior BWT for naive/replay/reactive.
- `results/sensitivity/per_stage_rewards.csv` — long-form per-task-per-stage reward matrix (audit).
- `results/sensitivity/continuous_rewards.csv` — per-stage trajectory.
- `results/sensitivity/demand_diagnostics.csv` — per-checkpoint drift magnitude (SMD/KS).
- `media/continuous_forgetting.png` — mean-prior BWT curves with bootstrap-CI ribbons.
- `media/cost_vs_slo.png` — Pareto plot with all 5 policies.
- `media/episode_rollout_comparison.png` — naive vs. replay on the same final-checkpoint demand.
- `media/replay_ratio_ablation.png` — mix-ratio sweep on the linear mapping.

The synthetic Phase 1 sanity run is in [`results/phase1/`](results/phase1/) and still works
via `make phase1`; it confirms reactive beats static-median on SLO on a bursty regime, and
is unrelated to the sensitivity numbers below.

**Primary Evaluation: Catastrophic Forgetting vs. Adaptation**

Two metrics are reported per mapping, both as signed backward transfer (positive = no
forgetting). The primary number is the **mean BWT across all prior tasks** —
`mean_i<k (R_{i, final} − R_{i, i})` — which is the standard continual-learning BWT and
uses every prior task, not just task 1. The legacy **Task-1 BWT** is retained as a
"first-task retention" secondary number.

All numbers reflect the latest local run (`seeds 7..11`, 5 seeds). Each mean is bracketed
by a 10k-resample percentile bootstrap 95% CI. The `Wilcoxon p` column is a two-sided
paired signed-rank test between per-seed naive and replay BWTs at the final stage. The
`Reactive` row is a pipeline sanity check — the reactive autoscaler is stateless and so
its BWT is exactly 0 by construction; any drift from zero would indicate a bug in the
eval harness, not in the policies.

See [`results/sensitivity/summary.md`](results/sensitivity/summary.md) for the autogenerated
table and [`results/sensitivity/per_seed_bwt.csv`](results/sensitivity/per_seed_bwt.csv)
for raw per-seed numbers. The headline table is reproduced here:

<!-- BWT_RESULTS_TABLE_START -->
| Mapping | Naive Task-1 BWT (95% CI) | Replay Task-1 BWT (95% CI) | Naive mean-prior BWT | Replay mean-prior BWT | Reactive (sanity) | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- |
| linear    | +5.25  [−221.86, +190.33] | +251.25 [+122.51, +426.04] | −21.11 [−66.67, +29.55] | **+90.08 [+58.54, +130.29]** | +0.00 | 0.312 |
| convex    | **−217.67 [−421.33, −68.24]** | **+206.67 [+89.98, +325.31]** | **−88.59 [−121.05, −56.13]** | **+52.16 [+18.20, +86.13]** | +0.00 | 0.062 |
| threshold | −127.86 [−340.53, +73.27]  | +134.86 [−188.47, +422.30] | −38.74 [−96.91, +19.43] | +45.13 [−35.76, +121.29] | +0.00 | 0.125 |

**Bold** entries are 95%-CI-significantly different from 0. Reactive is the sanity row;
its BWT is 0 by construction since the reactive autoscaler does not learn.

**Honest summary of what these numbers say:**

- **Replay is directionally better than Naive on both BWT metrics in all three mappings.** Replay's mean-prior BWT is positive in every mapping; Naive's is negative or near-zero.
- **The convex mapping is the cleanest result.** Both BWT metrics are CI-significant: Naive is significantly *negative* (catastrophic forgetting across prior tasks), Replay is significantly *positive* (net positive transfer). Wilcoxon paired test on the per-seed Task-1 BWT is p = 0.062 — marginal at α = 0.05 with n = 5.
- **Linear mean-prior BWT for Replay is significantly positive** [+58.54, +130.29] — replay reliably retains performance across all prior tasks even when the Task-1-only number is noisier.
- **Threshold mapping has the widest spread.** Replay direction is positive but neither CI excludes 0; this is the most variance-bound result, consistent with the threshold mapping's higher per-stage demand variance.
- **No Wilcoxon test at n = 5 reaches p < 0.05** on Task-1 BWT, so the *strict* statistical-significance claim is: "Replay shows directionally consistent improvement on every mapping and both metrics, with mean-prior BWT CIs significantly above zero for linear and convex. Task-1-only paired test does not reach p < 0.05 with this seed budget; n ≥ 8 would tighten it." Run with `--seed-count 8` to retest.
<!-- BWT_RESULTS_TABLE_END -->

**Per-checkpoint demand drift** (from `results/sensitivity/demand_diagnostics.csv`):

<!-- DRIFT_TABLE_START -->
| Checkpoint | Demand mean | Demand std | SMD vs ckpt 1 | KS vs ckpt 1 | Different? |
| --- | --- | --- | --- | --- | --- |
| 1   | 2.907 | 0.518 | —      | —     | — |
| 25  | 1.829 | 0.149 | −2.828 | 0.986 | yes |
| 50  | 2.549 | 0.944 | −0.469 | 0.549 | yes |
| 75  | 1.570 | 0.309 | −3.134 | 0.943 | yes |
| 100 | 2.086 | 0.475 | −1.651 | 0.672 | yes |
| 125 | 2.663 | 0.303 | −0.574 | 0.370 | yes |

Every post-baseline checkpoint passes the configured drift gate (`SMD ≥ 0.5` or
`KS ≥ 0.30`). The run is otherwise refused with a configuration error; this gate prevents
the forgetting metric from being reported on a curriculum that doesn't actually shift.
<!-- DRIFT_TABLE_END -->

**Secondary Metric — Cost vs. SLO** ([`media/cost_vs_slo.png`](media/cost_vs_slo.png)):
the Pareto plot evaluates all of `static_median`, `static_p95`, `reactive_threshold`,
`ppo_naive` (after drift), and `ppo_replay` (after drift) on the SAME held-out trace
(the final Azure checkpoint). PPO variants are averaged across the same seeds as the
BWT table, with error bars on both axes.

Honest read on this run: on the final checkpoint, **the static policies and reactive
autoscaler dominate the PPO variants on this proxy** — both static policies hit 0% SLO
violations at cost ~18, reactive 0% at ~31, while PPO + Replay sits at ~27% SLO at cost
~57 and PPO + Naive at ~43% SLO at cost ~16 (lowest cost but worst SLO; on the Pareto
frontier in cost terms only). This is *expected and intentionally not papered over* —
the contribution of this benchmark is forgetting measurement under drift, not a claim
that PPO beats reactive on the cost/SLO snapshot. See the Limitations section.

**Side-by-side rollout** ([`media/episode_rollout_comparison.png`](media/episode_rollout_comparison.png)):
naive vs. replay on the same final-checkpoint demand trace, with reactive overlay and
per-panel SLO-violation count and total task-cost. On the linear-mapping seed-7 final
checkpoint, naive converges to a higher constant capacity (0 violations, 90 task-units)
while replay runs at a lower capacity (4 violations, 86 task-units). Neither is "right"
— this panel exists to show concrete behavior, not to imply one is better on the cost
proxy.

**Replay-mix-ratio ablation** ([`media/replay_ratio_ablation.png`](media/replay_ratio_ablation.png),
[`results/replay_ratio_ablation/summary.md`](results/replay_ratio_ablation/summary.md)):
final-stage BWT as a function of `replay_mix_ratio` on the linear mapping (3 seeds, 4
ratios). Headline numbers:

| Mix Ratio | Task-1 BWT (95% CI) | Mean-prior BWT (95% CI) |
| --- | --- | --- |
| 0.00 | +132.10 [+75.79, +197.51] | +16.07 [−12.00, +45.92] |
| 0.25 | +159.57 [+59.26, +232.30] | +79.18 [+68.05, +98.98] |
| 0.50 | +156.51 [+96.70, +192.95] | +68.25 [+61.89, +77.25] |
| 0.75 | +153.83 [+97.32, +191.09] | +90.85 [+62.68, +127.82] |

The 0.00 point holds the replay infrastructure constant while turning off the actual
replay envs — it's the within-script sanity check. Mean-prior BWT jumps from ~0 to
~70-90 as soon as any non-zero replay is mixed in, then flattens. The current default
`replay_mix_ratio = 0.25` captures most of the benefit; 0.5 and 0.75 don't materially
improve mean-prior BWT on this mapping at n = 3 seeds. Default kept as-is.

## Methodology

### Statistical Conventions

- **Bootstrap CIs** in every plot and table use 10k percentile resamples by default (`--bootstrap-resamples`).
- **Wilcoxon paired** signed-rank tests compare per-seed Naive vs. Replay final-stage BWT, two-sided, with `α = 0.05` (`--alpha`).
- **Seeds** are a deterministic range starting at the PPO config seed (`seed = 7`), default 5 seeds (`seeds 7..11`), CLI-configurable via `--seeds` or `--seed-count`. Don't reuse hand-picked seeds — pick a deterministic range and document it.
- **Reactive baseline row** in `summary.csv` is the eval pipeline's sanity check; its BWT must be exactly 0. If it isn't, fix the eval before trusting the PPO numbers.

### Environment Design (`DriftScaleEnv`)

A custom Gymnasium environment simulating workload windows. The agent observes recent utilization features (normalized to `[-1, 1]`) and outputs a discrete action to scale tasks by `{-2, -1, 0, +1, +2}`, utilizing Stable Baselines3's `MaskablePPO` to prevent invalid boundary actions.

**Reward Function:**

```python
reward = -(
    (task_count * cost_per_task) 
  + (slo_weight * max(0, demand - capacity) / max(demand, eps))
  + (action_weight * abs(new_tasks - old_tasks))
  + (overprovision_weight * max(0, capacity - demand) / max(capacity, eps))
)

```

### Trace-to-Demand Mapping (Sensitivity Analysis)

Because Azure VM CPU traces represent utilization rather than direct service request counts, DriftScale maps raw CPU to an abstract "demand" signal. To ensure results aren't artifacts of a single mapping choice, the headline forgetting metrics are evaluated across three distinct variants:

1. **Linear (Default):** `demand ∝ sum(cpu)`
2. **Convex:** `demand ∝ sum(cpu)^1.5` (Models severe contention/queueing)
3. **Threshold:** `demand = sum(cpu) + α · indicator(any vm > 0.9)` (Captures tail spikes)

*Results are only considered robust if the forgetting reduction holds directionally across all three mappings.*

### Baselines

1. **Reactive Threshold:** The reference point (Scale up if CPU > 70% for N steps, down if < 30% for M steps).
2. **Static p95 Overprovisioned:** Upper bound on SLO safety.
3. **Static Median:** Lower bound on cost.
4. **Naive Fine-Tuning (Vanilla PPO):** The catastrophic forgetting baseline.
5. **PPO Trained from Scratch:** Measures adaptation speed.

## AWS Demo

Planned for a later phase. Phase 1 intentionally does not deploy cloud resources.

To validate that the RL controller can safely interface with cloud APIs, a small-scale, safety-constrained live demo is included.

* **Stack:** Terraform, ECS Fargate, Docker, FastAPI, Application Load Balancer, `k6`, CloudWatch, `boto3`.
* **Guardrails:** Strict controller limits (`min_tasks=1`, `max_tasks=6`, 60-second cooldowns, max scale delta of 1). AWS Budgets alarms prevent overspending.
* **Execution:** `k6` replays a time-compressed Azure workload regime against the FastAPI `/cpu` burn endpoint, while the local RL controller reads CloudWatch metrics and securely updates the ECS `desiredCount`.

## Limitations

* **Simulation Fidelity:** Azure VM CPU traces are not raw service request traces. The mapping loses fidelity for true latency and queueing dynamics, which is why the §5.5 sensitivity analysis is strictly enforced.
* **Cohort Construction:** The forgetting benchmark now uses per-checkpoint dense VM cohorts to preserve workload-composition drift. Larger VM counts can central-limit-smooth the aggregate demand, so every run saves demand mean/std/p95 plus SMD/KS drift diagnostics before the forgetting metrics are interpreted.
* **Induced vs. natural drift:** The workload regimes used here are induced by selecting *behaviorally distinct VM cohorts per checkpoint* — i.e. the drift is structurally introduced through cohort selection, not observed in a single time series of a single fleet. Production drift in a real auto-scaled service may be substantially gentler (a slowly shifting distribution within the same cohort) or substantially sharper (sudden migration to a new VM family, new deployment region, etc.). The benchmark is calibrated for "measurable shift" via the SMD/KS gate; it is not a stand-in for any specific production trace.
* **Drift Paradigms:** Real cloud drift is continuous. While this project tests a multi-checkpoint regime built from per-checkpoint dense VM cohorts, interleaved replay still fundamentally assumes identifiable task boundaries.
* **EWC scoped out:** Elastic Weight Consolidation was originally planned as a second anti-forgetting method, but actor-critic Fisher-information regularization (especially across value-head shifts) is fragile and would have required a multi-day stabilization effort. It is left as future work; the baseline-of-record comparison here is naive fine-tuning versus interleaved replay.
* **SLO Proxy:** The simulation uses capacity-vs-demand as a proxy for SLO violations. Only the live AWS demo generates real `p95` latency metrics.
* **Reactive scaling is the reference, not the rival:** Whether RL strictly beats reactive scaling on cost/SLO is a secondary finding. The core contribution is the measurement of online adaptation.

## Lessons Learned

* **Start with simulator invariants:** Phase 1 focuses on normalized observations, explicit reward components, and tests before training any policy.
* **Mask invalid scale actions early:** Boundary actions are exposed through `action_masks()` so later MaskablePPO training does not learn from silently clamped actions.
* **Reactive is a serious reference:** On the bursty synthetic regime, reactive autoscaling materially reduces SLO violations versus static median while carrying the expected higher task cost.
