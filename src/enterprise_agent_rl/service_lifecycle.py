"""Small, dependency-free primitives for bounded local service supervision.

The launcher remains responsible for the service-specific HTTP contract and
for persistence.  This module owns only the monotonic budget, signal state,
child-process classification, and bounded process-group teardown so those
rules can be exercised with CPU-only subprocesses.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field


class LifecycleError(RuntimeError):
    """Base class for expected supervisor termination paths."""


class DeadlineExceeded(LifecycleError):
    """The single run-global work deadline was reached."""

    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(f"deadline_exceeded:{stage}")


class SupervisorSignal(LifecycleError):
    """An operator or scheduler signal interrupted the supervised run."""

    def __init__(self, signum: int):
        self.signum = signum
        self.exit_code = 128 + signum
        super().__init__(f"signal:{signum}")


class ServiceProcessExited(LifecycleError):
    """A child owned by the supervisor exited before normal teardown."""

    def __init__(self, label: str, returncode: int | None):
        self.label = label
        self.returncode = returncode
        super().__init__(f"{label}_exited:{returncode}")


@dataclass
class LifecycleClock:
    """One monotonic work deadline plus a separate bounded cleanup reserve."""

    soft_cutoff_s: float = 1320.0
    cleanup_reserve_s: float = 60.0
    started_at: float = field(default_factory=time.monotonic)
    monotonic_fn: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if self.soft_cutoff_s <= 0:
            raise ValueError("soft_cutoff_s_must_be_positive")
        if self.cleanup_reserve_s < 0:
            raise ValueError("cleanup_reserve_s_must_be_non_negative")
        self.run_deadline = self.started_at + self.soft_cutoff_s
        self.cleanup_deadline = self.run_deadline + self.cleanup_reserve_s

    def now(self) -> float:
        return self.monotonic_fn()

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self.now() - self.started_at)

    @property
    def remaining_s(self) -> float:
        return self.run_deadline - self.now()

    @property
    def cleanup_remaining_s(self) -> float:
        return self.cleanup_deadline - self.now()

    def check(self, stage: str) -> None:
        if self.remaining_s <= 0:
            raise DeadlineExceeded(stage)

    def timeout(self, stage_limit_s: float, stage: str) -> float:
        """Return a blocking timeout bounded by the one run-global deadline."""

        if stage_limit_s <= 0:
            raise ValueError("stage_limit_s_must_be_positive")
        remaining = self.remaining_s
        if remaining <= 0:
            raise DeadlineExceeded(stage)
        return min(stage_limit_s, remaining)


@dataclass
class SignalController:
    """Signal state shared by the handler and blocking supervisor helpers."""

    signum: int | None = None
    cleaning: bool = False

    def request(self, signum: int) -> None:
        if self.signum is None:
            self.signum = signum

    def raise_if_requested(self) -> None:
        if self.signum is not None and not self.cleaning:
            raise SupervisorSignal(self.signum)


def stop_process_group(
    process: subprocess.Popen | None,
    grace_s: float = 15.0,
    *,
    cleanup_deadline: float | None = None,
    evidence: dict | None = None,
) -> int | None:
    """TERM, bounded wait, KILL, and bounded wait for one owned process group.

    Children are created with ``start_new_session=True`` by the launcher, so
    the process PID is also the private process-group ID.  No process lookup
    or broad-pattern kill is used here.
    """

    if process is None:
        return None
    evidence = evidence if evidence is not None else {}
    evidence.setdefault("pid", process.pid)
    evidence.setdefault("process_group_owned", True)
    if process.poll() is not None:
        evidence.update({"already_exited": True, "returncode": process.returncode})
        return process.returncode

    def bounded_wait(timeout_s: float) -> bool:
        try:
            process.wait(timeout=max(0.01, timeout_s))
            return True
        except subprocess.TimeoutExpired:
            return False

    try:
        os.killpg(process.pid, signal.SIGTERM)
        evidence["term_sent"] = True
    except ProcessLookupError:
        evidence["term_process_group_missing"] = True
    grace = max(0.01, grace_s)
    if cleanup_deadline is not None:
        grace = min(grace, max(0.01, cleanup_deadline - time.monotonic()))
    if bounded_wait(grace):
        evidence.update({"kill_sent": False, "wait_completed": True})
        return process.returncode

    try:
        os.killpg(process.pid, signal.SIGKILL)
        evidence["kill_sent"] = True
    except ProcessLookupError:
        evidence["kill_process_group_missing"] = True
    remaining = 5.0
    if cleanup_deadline is not None:
        remaining = min(remaining, max(0.01, cleanup_deadline - time.monotonic()))
    evidence["wait_after_kill_timeout_s"] = remaining
    evidence["wait_completed"] = bounded_wait(remaining)
    evidence["returncode"] = process.returncode
    return process.returncode
