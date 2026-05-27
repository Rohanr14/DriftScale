from datetime import UTC, datetime, timedelta

from driftscale.aws.safety import EcsSafetyWrapper, SafetyConfig


def test_safety_clamps_large_policy_delta() -> None:
    wrapper = EcsSafetyWrapper(SafetyConfig(max_scale_delta=1))

    decision = wrapper.apply(
        current_count=3,
        proposed_delta=2,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert decision.desired_count == 4
    assert decision.bounded_delta == 1
    assert decision.should_update
    assert decision.intervention == "max_delta"


def test_safety_blocks_scaling_during_cooldown() -> None:
    wrapper = EcsSafetyWrapper(SafetyConfig(cooldown_seconds=60))
    first = datetime(2026, 1, 1, tzinfo=UTC)
    wrapper.apply(current_count=2, proposed_delta=1, now=first)

    decision = wrapper.apply(
        current_count=3,
        proposed_delta=-1,
        now=first + timedelta(seconds=30),
    )

    assert decision.desired_count == 3
    assert not decision.should_update
    assert decision.intervention == "cooldown"


def test_safety_enforces_task_bounds() -> None:
    wrapper = EcsSafetyWrapper(SafetyConfig(min_tasks=1, max_tasks=6))

    decision = wrapper.apply(
        current_count=1,
        proposed_delta=-1,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert decision.desired_count == 1
    assert not decision.should_update
    assert decision.intervention == "bounds"
