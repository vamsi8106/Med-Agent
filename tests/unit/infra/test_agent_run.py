from medagent.infra.agent_run import AgentRunTracker


def test_record_accumulates_tokens_across_steps() -> None:
    tracker = AgentRunTracker(max_tokens=1000, run_id="run-1")

    tracker.record("drug_safety", {"prompt_tokens": 100, "completion_tokens": 50})
    tracker.record("evidence", {"prompt_tokens": 200, "completion_tokens": 25})

    assert tracker.tokens_used == 375
    assert [s.status for s in tracker.steps] == ["completed", "completed"]


def test_record_with_no_usage_counts_zero_tokens() -> None:
    tracker = AgentRunTracker(max_tokens=1000, run_id="run-1")

    tracker.record("triage", usage=None)

    assert tracker.tokens_used == 0
    assert tracker.steps[0].tokens_used == 0


def test_over_budget_true_once_ceiling_reached() -> None:
    tracker = AgentRunTracker(max_tokens=100, run_id="run-1")

    assert tracker.over_budget() is False
    tracker.record("drug_safety", {"prompt_tokens": 60, "completion_tokens": 50})
    assert tracker.over_budget() is True


def test_skip_records_step_and_is_reported_as_skipped() -> None:
    tracker = AgentRunTracker(max_tokens=100, run_id="run-1")
    tracker.record("drug_safety", {"prompt_tokens": 100})

    tracker.skip("evidence")

    assert tracker.skipped_steps() == ["evidence"]
    assert tracker.steps[-1].detail == "token budget exhausted: 100/100"


def test_trace_summary_lists_every_step_with_status_and_tokens() -> None:
    tracker = AgentRunTracker(max_tokens=100, run_id="run-1")
    tracker.record("drug_safety", {"prompt_tokens": 100})
    tracker.skip("evidence")

    summary = tracker.trace_summary()

    assert "drug_safety=completed(100 tok)" in summary
    assert "evidence=skipped(0 tok)" in summary
