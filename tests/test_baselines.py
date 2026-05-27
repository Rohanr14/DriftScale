from driftscale.baselines.reactive import ReactiveAutoscaler
from driftscale.baselines.static import evaluate_static_policy, static_task_count
from driftscale.traces.synthetic import generate_synthetic_episode


def test_reactive_baseline_beats_static_median_on_slo_for_bursty_regime() -> None:
    episode = generate_synthetic_episode(regime="bursty", length=288, seed=7, noise=0.04)
    median_tasks = static_task_count(episode.demand, quantile=0.50)
    static_median = evaluate_static_policy(
        episode.demand,
        task_count=median_tasks,
        policy_name="static_median",
    )
    reactive = ReactiveAutoscaler(initial_tasks=4, scale_step=2).evaluate(episode.demand)

    assert reactive.slo_violation_rate < static_median.slo_violation_rate


def test_static_p95_trades_higher_cost_for_lower_slo_than_static_median() -> None:
    episode = generate_synthetic_episode(regime="bursty", length=288, seed=7, noise=0.04)
    median_tasks = static_task_count(episode.demand, quantile=0.50)
    p95_tasks = static_task_count(episode.demand, quantile=0.95)

    static_median = evaluate_static_policy(
        episode.demand,
        task_count=median_tasks,
        policy_name="static_median",
    )
    static_p95 = evaluate_static_policy(
        episode.demand,
        task_count=p95_tasks,
        policy_name="static_p95",
    )

    assert static_p95.slo_violation_rate <= static_median.slo_violation_rate
    assert static_p95.total_resource_cost > static_median.total_resource_cost

