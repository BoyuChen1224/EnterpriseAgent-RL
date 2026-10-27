"""Persistent allocated-GPU wall-clock budget and child-process guard."""

# 训练/评测运行的持久预算守卫：按进程内单调时间×GPU 数量累计，并独立限制调用次数。
# 此计时从 start_job 开始，不等于 Slurm 全部 allocation；最终资源报告应核对调度侧来源。
# 接入本守卫的 run 可跨尝试累计；不能据此宣称所有服务入口都已实现跨 Job 自动预算。

from __future__ import annotations

import contextlib
import fcntl
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .io import atomic_write_json, read_json


class BudgetError(RuntimeError):
    """A real computation must stop or must not be scheduled."""


class BudgetHistoryError(BudgetError):
    """The durable accounting record is missing, contradictory, or open."""


class BudgetSoftStop(BudgetError):
    """The soft deadline was reached; no new work may be scheduled."""


class BudgetHardStop(BudgetError):
    """The hard deadline or cleanup reserve was reached."""


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def time(self) -> float: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def time(self) -> float:
        return time.time()


@dataclass(frozen=True)
class BudgetSnapshot:
    consumed_gpu_seconds: float
    run_remaining_seconds: float
    job_remaining_seconds: float
    safe_work_seconds: float
    soft_stop: bool
    phase: str

    @property
    def recommended_slurm_seconds(self) -> int:
        return max(1, int(math.ceil(self.safe_work_seconds + 60)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "consumed_gpu_seconds": self.consumed_gpu_seconds,
            "run_remaining_seconds": self.run_remaining_seconds,
            "job_remaining_seconds": self.job_remaining_seconds,
            "safe_work_seconds": self.safe_work_seconds,
            "soft_stop": self.soft_stop,
            "phase": self.phase,
            "recommended_slurm_seconds": self.recommended_slurm_seconds,
        }


def _finite_nonnegative(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise BudgetHistoryError(f"invalid_budget_value:{label}:{value}")
    return result


# 同一 run 使用同一份状态文件，通过文件锁和原子写入维护累计 GPU 秒与调用账本。
class BudgetGuard:
    """Single-run accounting shared by every stage of the online loop.

    ``start_job`` is the point at which a Slurm allocation is actually being
    used.  Queue time never enters the state.  Every ``check`` accrues the
    elapsed allocated-GPU wall time and persists it atomically.  A process that
    dies while an attempt is open is intentionally not guessed at: the next
    process must call ``recover_interrupted`` with externally recorded elapsed
    time or the guard rejects new computation.
    """

    schema_version = "grpo-budget-v1"

    def __init__(
        self,
        state_path: Path,
        *,
        run_id: str,
        run_hard_seconds: int,
        job_hard_seconds: int,
        soft_seconds: int,
        cleanup_reserve_seconds: int = 60,
        gpu_count: int = 1,
        call_limits: dict[str, int] | None = None,
        clock: Clock | None = None,
    ) -> None:
        if gpu_count < 1:
            raise ValueError("gpu_count_must_be_positive")
        self.state_path = state_path
        self.lock_path = state_path.with_suffix(state_path.suffix + ".lock")
        self.run_id = run_id
        self.run_hard_seconds = int(run_hard_seconds)
        self.job_hard_seconds = int(job_hard_seconds)
        self.soft_seconds = int(soft_seconds)
        self.cleanup_reserve_seconds = int(cleanup_reserve_seconds)
        self.gpu_count = int(gpu_count)
        self.call_limits = {str(key): int(value) for key, value in (call_limits or {}).items()}
        if any(value < 1 for value in self.call_limits.values()):
            raise ValueError("call_limits_must_be_positive")
        self.clock = clock or SystemClock()
        self._validate_limits()
        self._ensure_state()

    def _validate_limits(self) -> None:
        if not 0 < self.soft_seconds < self.job_hard_seconds <= self.run_hard_seconds:
            raise ValueError("budget_limits_not_monotonic")
        if not 0 < self.cleanup_reserve_seconds < self.job_hard_seconds:
            raise ValueError("invalid_cleanup_reserve")

    @contextlib.contextmanager
    def _locked(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _initial_state(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "limits": {
                "run_hard_seconds": self.run_hard_seconds,
                "job_hard_seconds": self.job_hard_seconds,
                "soft_seconds": self.soft_seconds,
                "cleanup_reserve_seconds": self.cleanup_reserve_seconds,
                "gpu_count": self.gpu_count,
                "call_limits": self.call_limits,
            },
            "consumed_gpu_seconds": 0.0,
            "call_counters": {},
            "attempts": [],
            "events": [],
            "active_attempt": None,
            "status": "NOT_STARTED",
        }

    def _ensure_state(self) -> None:
        with self._locked():
            state = read_json(self.state_path)
            if state is None:
                atomic_write_json(self.state_path, self._initial_state())
                return
            self._validate_state(state)

    def _validate_state(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != self.schema_version:
            raise BudgetHistoryError("budget_schema_mismatch")
        if state.get("run_id") != self.run_id:
            raise BudgetHistoryError("budget_run_id_mismatch")
        limits = state.get("limits")
        if isinstance(limits, dict):
            limits = {**limits, "call_limits": limits.get("call_limits", {})}
        expected = {
            "run_hard_seconds": self.run_hard_seconds,
            "job_hard_seconds": self.job_hard_seconds,
            "soft_seconds": self.soft_seconds,
            "cleanup_reserve_seconds": self.cleanup_reserve_seconds,
            "gpu_count": self.gpu_count,
            "call_limits": self.call_limits,
        }
        if limits != expected:
            raise BudgetHistoryError("budget_limits_changed")
        consumed = _finite_nonnegative(state.get("consumed_gpu_seconds", -1), "consumed")
        if consumed > self.run_hard_seconds + 1e-6:
            raise BudgetHistoryError("budget_consumed_exceeds_run_hard")
        if not isinstance(state.get("attempts", []), list) or not isinstance(
            state.get("events", []), list
        ):
            raise BudgetHistoryError("budget_history_not_lists")
        counters = state.get("call_counters", {})
        if not isinstance(counters, dict) or any(
            not isinstance(value, int) or value < 0 for value in counters.values()
        ):
            raise BudgetHistoryError("budget_call_counters_invalid")
        active = state.get("active_attempt")
        if active is not None and not isinstance(active, dict):
            raise BudgetHistoryError("budget_active_attempt_invalid")
        closed_elapsed = 0.0
        for attempt in state.get("attempts", []):
            if not isinstance(attempt, dict):
                raise BudgetHistoryError("budget_attempt_invalid")
            closed_elapsed += _finite_nonnegative(
                attempt.get("elapsed_gpu_seconds", -1), "closed_attempt_elapsed"
            )
        active_elapsed = 0.0
        if active is not None:
            active_elapsed = _finite_nonnegative(
                active.get("elapsed_gpu_seconds", -1), "active_attempt_elapsed"
            )
        if closed_elapsed + active_elapsed > consumed + 1e-6:
            raise BudgetHistoryError("budget_elapsed_exceeds_consumed")
        if active is None and consumed > 1e-6 and not state.get("attempts"):
            raise BudgetHistoryError("budget_consumed_history_missing")

    def _load(self) -> dict[str, Any]:
        state = read_json(self.state_path)
        if not isinstance(state, dict):
            raise BudgetHistoryError("budget_state_missing")
        self._validate_state(state)
        return state

    def _save_event(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        state.setdefault("events", []).append(event)
        atomic_write_json(self.state_path, state)

    # 将上次核算后的时间差乘实际配置卡数，累计到当前尝试和整个 run。
    # 排队未开始不在此计时中，失败后已花的时间也不会自动清零。
    def _accrue(self, state: dict[str, Any], *, phase: str) -> float:
        active = state.get("active_attempt")
        if active is None:
            raise BudgetError("budget_job_not_started")
        now = self.clock.monotonic()
        last = _finite_nonnegative(active.get("last_accounted_monotonic", -1), "last_accounted")
        if now < last:
            raise BudgetHistoryError("clock_moved_backwards")
        elapsed = (now - last) * int(active.get("gpu_count", self.gpu_count))
        active["last_accounted_monotonic"] = now
        active["elapsed_gpu_seconds"] = (
            _finite_nonnegative(active.get("elapsed_gpu_seconds", 0), "attempt_elapsed") + elapsed
        )
        state["consumed_gpu_seconds"] = (
            _finite_nonnegative(state["consumed_gpu_seconds"], "consumed") + elapsed
        )
        state["last_phase"] = phase
        return elapsed

    def _snapshot(self, state: dict[str, Any], phase: str) -> BudgetSnapshot:
        active = state.get("active_attempt")
        elapsed_job = float(active.get("elapsed_gpu_seconds", 0.0)) if active else 0.0
        consumed = float(state["consumed_gpu_seconds"])
        run_remaining = max(0.0, self.run_hard_seconds - consumed)
        job_remaining = (
            max(0.0, self.job_hard_seconds - elapsed_job)
            if active is not None
            else float(self.job_hard_seconds)
        )
        safe = max(0.0, min(run_remaining, job_remaining) - self.cleanup_reserve_seconds)
        return BudgetSnapshot(
            consumed_gpu_seconds=consumed,
            run_remaining_seconds=run_remaining,
            job_remaining_seconds=job_remaining,
            safe_work_seconds=safe,
            soft_stop=consumed >= self.soft_seconds,
            phase=phase,
        )

    # 有未收尾的旧尝试时拒绝直接重开，要求先核实并恢复中断账本。
    def start_job(
        self, *, job_id: str | None = None, attempt_id: str | None = None
    ) -> BudgetSnapshot:
        with self._locked():
            state = self._load()
            if state.get("active_attempt") is not None:
                raise BudgetHistoryError("unfinished_previous_attempt_requires_explicit_recovery")
            if (
                float(state["consumed_gpu_seconds"]) + self.cleanup_reserve_seconds
                >= self.run_hard_seconds
            ):
                raise BudgetHardStop("no_budget_remaining_for_new_job")
            now = self.clock.monotonic()
            active = {
                "attempt_id": attempt_id or f"attempt-{os.getpid()}-{int(self.clock.time())}",
                "job_id": job_id,
                "gpu_count": self.gpu_count,
                "started_at_epoch": self.clock.time(),
                "started_monotonic": now,
                "last_accounted_monotonic": now,
                "elapsed_gpu_seconds": 0.0,
                "status": "RUNNING",
            }
            state["active_attempt"] = active
            state["status"] = "RUNNING"
            state["events"].append({"event": "job_started", **active})
            snapshot = self._snapshot(state, "setup")
            state["last_snapshot"] = snapshot.as_dict()
            atomic_write_json(self.state_path, state)
            return snapshot

    # 按显式恢复信息处理被中断的尝试，不把未知历史用量默认为零。
    def recover_interrupted(self, *, elapsed_gpu_seconds: float, reason: str) -> BudgetSnapshot:
        """Close an open attempt only with an explicit external elapsed record."""

        elapsed = _finite_nonnegative(elapsed_gpu_seconds, "recovery_elapsed")
        with self._locked():
            state = self._load()
            active = state.get("active_attempt")
            if active is None:
                raise BudgetHistoryError("no_interrupted_attempt_to_recover")
            expected = float(active.get("elapsed_gpu_seconds", 0.0))
            if elapsed < expected:
                raise BudgetHistoryError("recovery_elapsed_less_than_accounted")
            additional = elapsed - expected
            new_consumed = (
                _finite_nonnegative(state["consumed_gpu_seconds"], "consumed") + additional
            )
            if new_consumed > self.run_hard_seconds + 1e-6:
                raise BudgetHardStop("recovered_elapsed_exceeds_run_budget")
            state["consumed_gpu_seconds"] = new_consumed
            closed = {
                **active,
                "elapsed_gpu_seconds": elapsed,
                "status": "INTERRUPTED",
                "closed_at_epoch": self.clock.time(),
                "reason": reason,
            }
            state["attempts"].append(closed)
            state["events"].append({"event": "job_recovered", **closed})
            state["active_attempt"] = None
            state["status"] = "RECOVERED"
            snapshot = self._snapshot(state, "recovery")
            state["last_snapshot"] = snapshot.as_dict()
            atomic_write_json(self.state_path, state)
            return snapshot

    def check(self, phase: str, *, allow_soft: bool = False) -> BudgetSnapshot:
        with self._locked():
            state = self._load()
            self._accrue(state, phase=phase)
            snapshot = self._snapshot(state, phase)
            state["last_snapshot"] = snapshot.as_dict()
            atomic_write_json(self.state_path, state)
            if (
                snapshot.run_remaining_seconds <= self.cleanup_reserve_seconds
                or snapshot.job_remaining_seconds <= self.cleanup_reserve_seconds
            ):
                raise BudgetHardStop(f"budget_hard_stop:{phase}:{snapshot.as_dict()}")
            if snapshot.soft_stop and not allow_soft:
                raise BudgetSoftStop(f"budget_soft_stop:{phase}:{snapshot.as_dict()}")
            return snapshot

    def close_job(self, *, status: str, reason: str | None = None) -> BudgetSnapshot:
        with self._locked():
            state = self._load()
            if state.get("active_attempt") is None:
                raise BudgetHistoryError("job_already_closed")
            self._accrue(state, phase="cleanup")
            active = state["active_attempt"]
            closed = {
                **active,
                "status": status,
                "closed_at_epoch": self.clock.time(),
                "reason": reason,
            }
            state["attempts"].append(closed)
            state["events"].append({"event": "job_closed", **closed})
            state["active_attempt"] = None
            state["status"] = status
            snapshot = self._snapshot(state, "cleanup")
            state["last_snapshot"] = snapshot.as_dict()
            atomic_write_json(self.state_path, state)
            return snapshot

    def current(self, phase: str = "query") -> BudgetSnapshot:
        with self._locked():
            state = self._load()
            if state.get("active_attempt") is not None:
                self._accrue(state, phase=phase)
            snapshot = self._snapshot(state, phase)
            state["last_snapshot"] = snapshot.as_dict()
            atomic_write_json(self.state_path, state)
            return snapshot

    # 每次实际操作按名称累计并检查上限；model、worker、Verifier SQL 和 backward 分开计数。
    def record_call(self, name: str, count: int = 1) -> int:
        """Persist a physical call counter and enforce its configured ceiling."""

        if count < 0:
            raise ValueError("call_count_must_be_nonnegative")
        with self._locked():
            state = self._load()
            if state.get("active_attempt") is None:
                raise BudgetError("budget_job_not_started")
            self._accrue(state, phase=f"count:{name}")
            counters = state.setdefault("call_counters", {})
            new_value = int(counters.get(name, 0)) + int(count)
            limit = self.call_limits.get(name)
            if limit is not None and new_value > limit:
                state["status"] = "CALL_LIMIT_STOP"
                state["events"].append(
                    {
                        "event": "call_limit_exceeded",
                        "name": name,
                        "count": new_value,
                        "limit": limit,
                        "phase": f"count:{name}",
                    }
                )
                atomic_write_json(self.state_path, state)
                raise BudgetHardStop(f"call_limit_exceeded:{name}:{new_value}:{limit}")
            counters[name] = new_value
            state["events"].append(
                {"event": "call_recorded", "name": name, "count": new_value, "delta": count}
            )
            atomic_write_json(self.state_path, state)
            return new_value

    def _record_watchdog_stop(self, phase: str) -> None:
        """Persist a deadline request before the caller receives the exception."""

        with self._locked():
            state = self._load()
            if state.get("active_attempt") is None:
                return
            self._accrue(state, phase=f"watchdog:{phase}")
            state["status"] = "HARD_STOP_REQUESTED"
            state["active_attempt"]["termination_requested"] = True
            state["events"].append(
                {"event": "watchdog_stop_requested", "phase": phase, "at": self.clock.time()}
            )
            atomic_write_json(self.state_path, state)

    # 给耗时函数加 watchdog；具体子进程操作还需配合进程组回收，不能只靠循环间检查。
    @contextlib.contextmanager
    def supervise(self, phase: str):
        """Guard a synchronous production call with a real process deadline.

        The handler runs in the main thread, where Python can interrupt a blocking
        Python/C call.  The durable watchdog event is written before raising, so a
        terminated process cannot be mistaken for a zero-cost successful phase.
        Child process groups remain supervised by ``run_subprocess``.
        """

        self.check(phase, allow_soft=True)
        snapshot = self.current(f"watchdog-arm:{phase}")
        delay = min(snapshot.run_remaining_seconds, snapshot.job_remaining_seconds)
        delay -= self.cleanup_reserve_seconds
        if delay <= 0:
            raise BudgetHardStop(f"budget_hard_stop:{phase}")
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)

        def handler(_signum, _frame):
            self._record_watchdog_stop(phase)
            raise BudgetHardStop(f"watchdog_hard_stop:{phase}")

        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, max(0.01, delay))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])

    def run_callable(self, function, *, phase: str):
        """Run one setup/generate/update/worker call under the watchdog."""

        with self.supervise(phase):
            result = function()
        self.check(f"{phase}:complete", allow_soft=True)
        return result

    # 监督子进程并在截止/信号时收回进程组，给日志和状态收尾保留预算。
    def run_subprocess(
        self,
        command: Sequence[str],
        *,
        phase: str,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        poll_seconds: float = 0.1,
    ) -> subprocess.CompletedProcess[str]:
        """Run a phase in its own process group and terminate it at the deadline."""

        self.check(phase)
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stopped = False
        try:
            while process.poll() is None:
                try:
                    self.check(phase, allow_soft=True)
                except BudgetError:
                    stopped = True
                    self._terminate_process_group(process)
                    raise
                time.sleep(min(max(poll_seconds, 0.01), 1.0))
            stdout, stderr = process.communicate()
            if stopped:
                raise BudgetHardStop(f"child_terminated:{phase}")
            self.check(phase, allow_soft=True)
            return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)
        finally:
            if process.poll() is None:
                self._terminate_process_group(process)

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 2.0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                process.wait(timeout=1)
