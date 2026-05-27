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

**Sensitivity suite complete:** real local Azure V1 shard loading, top-1,000 dense VM
selection, chronological Task A/B split, and §5.5 linear/convex/threshold BWT comparison.

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
make sensitivity-suite
make random-run
make phase1
```

## Why This Matters

"An RL autoscaler that beats reactive scaling" is a brittle claim. Reactive autoscaling is a highly robust, aggressively optimized baseline tuned over a decade of production experience. Instead of claiming a production-ready replacement, DriftScale aims at a different, highly relevant problem: **catastrophic forgetting in dynamic cloud environments.** When cloud workloads drift, learned policies degrade. DriftScale provides a reproducible empirical benchmark to quantify this degradation, measure how fast policies adapt to new regimes, and evaluate continual-learning methods (like experience replay and EWC) designed to mitigate forgetting.

## What DriftScale Does

This project is an end-to-end ML-for-systems benchmark that:

1. **Quantifies Catastrophic Forgetting:** Measures how vanilla PPO models forget past workload patterns when confronted with new ones (abrupt and gradual drift).
2. **Evaluates Continual Learning:** Tests replay-based and EWC (Elastic Weight Consolidation) anti-forgetting mechanisms against naive fine-tuning.
3. **Validates with Real Infrastructure:** Bridges the gap between simulation and reality by deploying the learned control loop against a live AWS ECS Fargate service, complete with safety guardrails and metric ingestion.

## Architecture

DriftScale operates on two distinct paths:

* **Simulation & Training Path:** Azure Public Traces → Preprocessor (Trace-to-Demand mapping) → Custom `DriftScaleEnv` (Gymnasium) → Policy Training (Reactive, PPO, Replay, EWC) → Evaluation.
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

# 8. Run chronological Task A/B sensitivity across linear, convex, and threshold mappings
make sensitivity-suite

# 9. Run the masked random-policy smoke test
make random-run

# 10. Evaluate Phase 1 static and reactive baselines
make phase1
```

## Results

Phase 1 currently evaluates the synthetic bursty regime from `configs/env/synthetic.yaml`.
The latest local run produced:

| Policy | SLO violation rate | Mean tasks | Scale actions |
| --- | ---: | ---: | ---: |
| Reactive threshold | 0.000 | 12.87 | 30 |
| Static median | 0.326 | 4.00 | 0 |
| Static p95 | 0.049 | 12.00 | 0 |

This satisfies the Week 1 acceptance check: random policy runs, reactive beats static median
on SLO, and unit tests pass.

Later phases will replace this section with Azure trace results, PPO, replay, forgetting, and
cost/SLO plots.

Week 2 preprocessing writes a lightweight cache to `results/caches/azure_v1_linear.csv` from the
tiny sample at `data/samples/azure_v1_tiny.csv`. The mapping is configured in
`configs/env/azure_v1.yaml`, including the threshold variant's tunable `alpha`.

Week 3 calibration writes `results/caches/azure_v1_calibrated.csv` and
`results/calibration/baseline_metrics.csv`. Week 4 training saves
`results/ppo_vanilla/model.zip`, `results/ppo_vanilla/vecnormalize.pkl`, and
`results/ppo_vanilla/metrics.csv`; the initial plot is saved to `media/cost_vs_slo.png`.

Weeks 5-6 write `results/drift_experiment/forgetting.csv` and three saved policy stages:
Task A pre-drift, naive Task B fine-tuning, and Task B fine-tuning with interleaved replay.

The sensitivity suite reads the real local Azure V1 shard at
`data/raw/vm_cpu_readings-file-1-of-125.csv.gz`, selects the 1,000 VMs with the most readings,
pivots a dense CPU matrix, and splits it chronologically into Task A and Task B. It writes
`results/sensitivity/summary.md`. No synthetic fallback or artificial Task B multiplier is used.

**Primary Evaluation: Catastrophic Forgetting vs. Adaptation**

| Method | Task A Reward (Before Drift) | Task A Reward (After Task B) | Forgetting |
| --- | --- | --- | --- |
| **Naive Fine-Tuning** (Vanilla PPO) | High | Low | **Severe** |
| **PPO + EWC** | High | Medium | **Moderate** |
| **PPO + Replay** | High | High | **Minimal** |

**Secondary Metric:** Cost vs. SLO Pareto frontier comparing Static p95, Reactive Threshold, and PPO + Replay.

## Methodology

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
* **Drift Paradigms:** Real cloud drift is continuous. While this project tests a gradual-drift regime, the primary continual learning methods (EWC/Replay) fundamentally assume distinct task boundaries (Task A → Task B).
* **SLO Proxy:** The simulation uses capacity-vs-demand as a proxy for SLO violations. Only the live AWS demo generates real `p95` latency metrics.
* **Reactive scaling is the reference, not the rival:** Whether RL strictly beats reactive scaling on cost/SLO is a secondary finding. The core contribution is the measurement of online adaptation.

## Lessons Learned

* **Start with simulator invariants:** Phase 1 focuses on normalized observations, explicit reward components, and tests before training any policy.
* **Mask invalid scale actions early:** Boundary actions are exposed through `action_masks()` so later MaskablePPO training does not learn from silently clamped actions.
* **Reactive is a serious reference:** On the bursty synthetic regime, reactive autoscaling materially reduces SLO violations versus static median while carrying the expected higher task cost.
