from __future__ import annotations

import sys

import pytest

from enterprise_agent_rl.grpo.budget import (
    BudgetGuard,
    BudgetHardStop,
    BudgetHistoryError,
    BudgetSoftStop,
)
from enterprise_agent_rl.grpo.io import atomic_write_json, read_json


class FakeClock:
    def __init__(self) -> None:
        self.monotonic_value = 0.0
        self.epoch_value = 1_700_000_000.0

    def monotonic(self) -> float:
        return self.monotonic_value

    def time(self) -> float:
        return self.epoch_value + self.monotonic_value

    def advance(self, seconds: float) -> None:
        self.monotonic_value += seconds


def make_guard(path, clock, *, run=3900, job=3600, soft=3300, reserve=60):
    return BudgetGuard(
        path,
        run_id="budget-test",
        run_hard_seconds=run,
        job_hard_seconds=job,
        soft_seconds=soft,
        cleanup_reserve_seconds=reserve,
        clock=clock,
    )


def test_persistent_balance_and_explicit_recovery(tmp_path):
    path = tmp_path / "budget-state.json"
    clock = FakeClock()
    pending = make_guard(path, clock)
    assert pending.current().consumed_gpu_seconds == 0

    pending.start_job(job_id="job-a")
    clock.advance(17 * 60 + 40)
    snapshot = pending.current("after-17m40s")
    assert snapshot.consumed_gpu_seconds == pytest.approx(1060.0)
    assert snapshot.run_remaining_seconds == pytest.approx(47 * 60 + 20)
    assert snapshot.recommended_slurm_seconds == 2540

    pending.recover_interrupted(elapsed_gpu_seconds=1060.0, reason="simulated_job_failure")
    resumed = make_guard(path, clock)
    resumed.start_job(job_id="job-b")
    assert resumed.current().consumed_gpu_seconds == pytest.approx(1060.0)
    clock.advance(10)
    resumed.close_job(status="COMPLETED")
    state = read_json(path)
    assert state["consumed_gpu_seconds"] == pytest.approx(1070.0)
    assert len(state["attempts"]) == 2
    assert state["attempts"][0]["status"] == "INTERRUPTED"


def test_soft_hard_cutoffs_and_open_attempt_rejection(tmp_path):
    path = tmp_path / "budget-state.json"
    clock = FakeClock()
    guard = make_guard(path, clock, run=100, job=100, soft=6, reserve=2)
    guard.start_job(job_id="cutoff")
    with pytest.raises(BudgetHistoryError, match="unfinished_previous_attempt"):
        make_guard(path, clock, run=100, job=100, soft=6, reserve=2).start_job(job_id="duplicate")

    clock.advance(7)
    with pytest.raises(BudgetSoftStop):
        guard.check("soft-boundary")
    guard.close_job(status="SOFT_STOP")

    hard_path = tmp_path / "hard-state.json"
    hard = make_guard(hard_path, clock, run=100, job=100, soft=80, reserve=2)
    hard.start_job(job_id="hard")
    clock.advance(99)
    with pytest.raises(BudgetHardStop):
        hard.check("hard-boundary", allow_soft=True)
    hard.close_job(status="HARD_STOP")


def test_in_progress_child_is_terminated_at_hard_deadline(tmp_path):
    path = tmp_path / "budget-state.json"
    guard = BudgetGuard(
        path,
        run_id="child-test",
        run_hard_seconds=4,
        job_hard_seconds=4,
        soft_seconds=3,
        cleanup_reserve_seconds=1,
    )
    guard.start_job(job_id="child")
    with pytest.raises(BudgetHardStop, match="budget_hard_stop"):
        guard.run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            phase="child-phase",
            poll_seconds=0.05,
        )
    state = read_json(path)
    elapsed = state["active_attempt"]["elapsed_gpu_seconds"]
    guard.recover_interrupted(elapsed_gpu_seconds=elapsed, reason="child_terminated_at_deadline")
    assert read_json(path)["active_attempt"] is None


def test_missing_or_contradictory_history_is_rejected(tmp_path):
    path = tmp_path / "invalid-state.json"
    atomic_write_json(path, {"schema_version": "wrong", "run_id": "budget-test"})
    with pytest.raises(BudgetHistoryError, match="budget_schema_mismatch"):
        make_guard(path, FakeClock())
