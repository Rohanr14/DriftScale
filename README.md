# DriftScale

**DriftScale benchmarks how reinforcement learning (RL) autoscaling policies degrade and recover under workload drift, using public Azure VM traces with reactive autoscaling as a reference baseline and an AWS ECS Fargate control-loop validation.**

## Demo

*(Placeholder: Loom link coming soon)*

> **Watch the DriftScale Demo:** A 5-minute walkthrough of the architecture, simulation loop, forgetting metrics, and a live validation run on AWS ECS.

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

This project is built with Python 3.11+ and uses `uv` (or `poetry`) for dependency management.

```bash
# 1. Install dependencies
make setup

# 2. Run test suite
make test

# 3. Preprocess the Azure traces (generates episodes based on configs)
make preprocess

# 4. Train the baseline (Vanilla PPO)
make train-baseline

# 5. Train with Replay continual learning
make train-replay

# 6. Evaluate all policies and generate metrics
make eval

# 7. Generate cost, SLO, and forgetting plots
make plots

```

For the AWS validation loop:

```bash
make deploy-demo  # Provisions Terraform infra (VPC, ALB, ECS)
make live-demo    # Runs the k6 load generator and RL control loop
make destroy-demo # Tears down all AWS resources

```

## Results

*(Note: These are target benchmarks; actual empirical results will be updated post-evaluation).*

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

* **Continual Learning Realities:** EWC + actor-critic proved brittle due to Fisher information matrices destabilizing the value function. Interleaved replay was theoretically simpler but drastically more reproducible.
* **Reward Hacking:** RL agents are ruthless. Initial penalty weights led to degenerate exploits (e.g., permanently collapsing to minimum tasks and eating the SLO penalty to save cost). Clipping penalties and observation normalization via `VecNormalize` were mandatory for stability.
* **Cloud API Lag:** Real CloudWatch metric ingestion has inherent delays. Translating simulated instantaneous state into an asynchronous, lagged AWS control loop required extensive tuning of the controller's safety cooldowns.
