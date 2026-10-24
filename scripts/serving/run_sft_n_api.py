#!/usr/bin/env python3
"""Run one bounded SFT-N model/API integration attempt.

The supervisor owns exactly the two process groups it creates.  It never
executes ``run_episode`` directly: the smoke goes through the Agent HTTP API,
which in turn uses LocalModelPolicy and the loopback model HTTP endpoint.
The explicit ``serve`` mode only keeps those services ready; it never schedules
benchmark work on its own.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from enterprise_agent_rl.service_config import (
    ServiceProfile,
    probe_free_loopback_port,
    sha256_file,
)
from enterprise_agent_rl.service_lifecycle import (
    DeadlineExceeded,
    LifecycleClock,
    ServiceProcessExited,
    SignalController,
    SupervisorSignal,
    stop_process_group,
)

_ACTIVE_SIGNAL_CONTROLLER: SignalController | None = None
DEFAULT_RUN_ROOT = Path("artifacts/service-runs")


def handle_supervisor_signal(signum, _frame) -> None:
    """Record and immediately interrupt the main-thread blocking operation."""

    if _ACTIVE_SIGNAL_CONTROLLER is None:
        raise SupervisorSignal(signum)
    _ACTIVE_SIGNAL_CONTROLLER.request(signum)
    if not _ACTIVE_SIGNAL_CONTROLLER.cleaning:
        raise SupervisorSignal(signum)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def json_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url,
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            **(headers or {}),
        },
        data=body,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            status = response.status
            raw = response.read(8 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        raw = exc.read(8 * 1024 * 1024)
        status = exc.code
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid_json_response:{url}:{status}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"non_object_response:{url}:{status}")
    return status, value


def choose_port(preferred: int, auto: bool) -> int:
    if auto:
        return probe_free_loopback_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
        except OSError as exc:
            raise RuntimeError(f"loopback_port_unavailable:{preferred}") from exc
    return preferred


def absolute_path_without_resolving(path: Path, *, base: Path) -> Path:
    """Make a relative file path absolute without following a symlink."""

    return path if path.is_absolute() else (base / path).absolute()


def resolve_python_executable(path: Path | None, *, root: Path) -> Path:
    """Preserve a venv entry-point symlink while making relative paths usable."""

    requested = path or (root / ".venv-gpu" / "bin" / "python")
    return absolute_path_without_resolving(requested, base=root)


def allocate_run_dir(
    profile: ServiceProfile,
    requested: Path | None,
    *,
    default_root: Path | None = None,
) -> Path:
    """Allocate one run without ever reusing an existing directory.

    An explicit path is a user-owned contract and therefore remains strict:
    an existing path is rejected.  The implicit path is a stable parent plus
    an atomically-created child, so an old run or a concurrent launcher cannot
    turn a new invocation into an overwrite.
    """

    if requested is not None:
        base = requested.resolve()
        if base.exists():
            raise FileExistsError(f"run_directory_exists:{base}")
        base.mkdir(parents=True)
        return base

    root = (default_root or profile.output_dir).absolute()
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(1000):
        candidate = root / f"run-{time.time_ns()}-{uuid4().hex[:12]}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("run_directory_allocation_exhausted")


def copy_read_only_database(profile: ServiceProfile, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    target = run_dir / "inputs" / "databases" / "train-1.sqlite3"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(profile.database_path, target)
    target.chmod(0o444)
    source_hash = sha256_file(profile.database_path)
    target_hash = sha256_file(target)
    if source_hash != profile.database_physical_sha256 or target_hash != source_hash:
        raise RuntimeError("copied_database_physical_sha256_mismatch")
    return target, {
        "source": str(profile.database_path),
        "copy": str(target),
        "physical_sha256": target_hash,
        "content_sha256": profile.database_content_sha256,
        "mode": oct(target.stat().st_mode & 0o777),
    }


def spawn_service(
    name: str,
    command: list[str],
    *,
    root: Path,
    run_dir: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen, dict[str, Any]]:
    stdout_path = run_dir / "process" / f"{name}.stdout.log"
    stderr_path = run_dir / "process" / f"{name}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = stdout_path.open("w")
    stderr = stderr_path.open("w")
    process = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        text=True,
    )
    stdout.close()
    stderr.close()
    return process, {
        "name": name,
        "pid": process.pid,
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "process_group_owned": True,
    }


def wait_for_ready(
    url: str,
    process: subprocess.Popen,
    timeout_s: float,
    label: str,
    *,
    clock: LifecycleClock | None = None,
    signal_controller: SignalController | None = None,
) -> dict[str, Any]:
    clock = clock or LifecycleClock(soft_cutoff_s=max(timeout_s, 1.0), cleanup_reserve_s=0)
    deadline = min(clock.run_deadline, time.monotonic() + timeout_s)
    last_error = "not_attempted"
    while True:
        if signal_controller is not None:
            signal_controller.raise_if_requested()
        remaining = min(deadline - time.monotonic(), clock.remaining_s)
        if remaining <= 0:
            if clock.remaining_s <= 0:
                raise DeadlineExceeded(f"{label}_ready")
            raise TimeoutError(f"{label}_ready_timeout:{last_error}")
        if process.poll() is not None:
            raise ServiceProcessExited(label, process.returncode)
        try:
            status, payload = json_request(url, timeout_s=min(30.0, remaining))
            if status == 200 and payload.get("status") in ("ready", "ok"):
                return payload
            last_error = f"status={status}:payload={payload}"
        except (SupervisorSignal, DeadlineExceeded):
            raise
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}:{exc}"
        sleep_s = min(0.25, max(0.01, remaining))
        time.sleep(sleep_s)


def write_service_snapshot(
    run_dir: Path,
    name: str,
    url: str,
    *,
    clock: LifecycleClock | None = None,
    signal_controller: SignalController | None = None,
) -> dict[str, Any]:
    clock = clock or LifecycleClock(soft_cutoff_s=30.0, cleanup_reserve_s=0)
    if signal_controller is not None:
        signal_controller.raise_if_requested()
    status, payload = json_request(url, timeout_s=clock.timeout(30.0, f"snapshot:{name}"))
    snapshot = {"url": url, "http_status": status, "payload": payload}
    write_json(run_dir / "health" / f"{name}.json", snapshot)
    return snapshot


def _read_panel(path: Path, task_id: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "task_id": task_id,
            "status": "unknown",
            "qualification": "technical_unknown",
            "evaluation_status": "technical_unknown",
            "terminal_reason": "not_scheduled",
        }
    try:
        value = json.loads(path.read_text())
    except (OSError, TypeError, json.JSONDecodeError):
        return {
            "task_id": task_id,
            "status": "unknown",
            "qualification": "technical_unknown",
            "evaluation_status": "technical_unknown",
            "terminal_reason": "panel_unreadable",
        }
    return value if isinstance(value, dict) else {
        "task_id": task_id,
        "status": "unknown",
        "qualification": "technical_unknown",
        "evaluation_status": "technical_unknown",
        "terminal_reason": "panel_invalid",
    }


def _read_jsonl_records(path: Path) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Read one run-local JSONL source, preserving missing/invalid provenance."""

    if not path.is_file():
        return None, "not_collected"
    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                return None, "record_not_object"
            records.append(value)
    except (OSError, TypeError, json.JSONDecodeError):
        return None, "source_unreadable"
    return records, None


def _read_final_trajectories(
    run_dir: Path,
) -> tuple[list[dict[str, Any]] | None, str | None, Path]:
    directory = run_dir / "api" / "trajectories"
    if not directory.is_dir():
        return None, "not_collected", directory
    paths = sorted(directory.glob("*.json"))
    records: list[dict[str, Any]] = []
    try:
        for path in paths:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                return None, "record_not_object", directory
            records.append(value)
    except (OSError, TypeError, json.JSONDecodeError):
        return None, "source_unreadable", directory
    return records, None, directory


def _count_evidence(
    value: int | None,
    source: Path,
    definition: str,
    *,
    reason: str | None = None,
    source_kind: str = "persisted_run_artifact",
) -> dict[str, Any]:
    return {
        "value": value,
        "source": str(source),
        "source_kind": source_kind,
        "definition": definition,
        "reason": reason,
    }


def collect_runtime_evidence(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Derive runtime counts once from durable, run-local evidence.

    Model events take precedence over trajectory ``model_attempts`` and SQL
    audit records take precedence over trajectory SQL steps.  The fallbacks
    are intentionally exclusive; they are never added to the primary source.
    A missing source is ``null`` rather than an inferred zero.
    """

    model_events_path = run_dir / "model" / "generations.jsonl"
    model_events, model_reason = _read_jsonl_records(model_events_path)
    trajectories, trajectory_reason, trajectory_dir = _read_final_trajectories(run_dir)

    if model_events is not None:
        model_calls = _count_evidence(
            len(model_events),
            model_events_path,
            (
                "one persisted model generation event per JSONL record; "
                "not added to trajectory attempts"
            ),
        )
    elif trajectories is not None:
        model_calls = _count_evidence(
            sum(len(value.get("model_attempts", [])) for value in trajectories),
            trajectory_dir,
            (
                "sum of persisted final-trajectory model_attempts because "
                "model event log is unavailable"
            ),
            source_kind="persisted_trajectory_fallback",
            reason="model_event_log_unavailable",
        )
    else:
        model_calls = _count_evidence(
            None,
            model_events_path,
            "one persisted model generation event per JSONL record",
            reason=model_reason or trajectory_reason or "no_model_source",
        )

    sql_audit_path = run_dir / "api" / "sql-audit.jsonl"
    sql_records, sql_reason = _read_jsonl_records(sql_audit_path)
    if sql_records is not None:
        business_sql_calls = _count_evidence(
            len(sql_records),
            sql_audit_path,
            (
                "one persisted SQLTool.execute audit record; schema calls are not "
                "counted by this audit"
            ),
        )
    elif trajectories is not None:
        business_sql_calls = _count_evidence(
            sum(
                sum(step.get("tool_name") == "sql" for step in value.get("steps", []))
                for value in trajectories
            ),
            trajectory_dir,
            "sum of final-trajectory SQL tool steps because SQL audit is unavailable",
            source_kind="persisted_trajectory_fallback",
            reason="sql_audit_unavailable",
        )
    else:
        business_sql_calls = _count_evidence(
            None,
            sql_audit_path,
            "one persisted SQLTool.execute audit record",
            reason=sql_reason or trajectory_reason or "no_sql_source",
        )

    if trajectories is not None:
        verifier_calls = _count_evidence(
            sum(value.get("reward") is not None for value in trajectories),
            trajectory_dir,
            (
                "one independent benchmark verifier call for each final trajectory "
                "with non-null reward"
            ),
        )
    else:
        verifier_calls = _count_evidence(
            None,
            trajectory_dir,
            (
                "one independent benchmark verifier call for each final trajectory "
                "with non-null reward"
            ),
            reason=trajectory_reason or "no_trajectory_source",
        )
    return {
        "model_calls": model_calls,
        "business_sql_calls": business_sql_calls,
        "verifier_calls": verifier_calls,
    }


def finalize_results(
    run_dir: Path,
    task_ids: tuple[str, ...] | list[str],
    *,
    mode: str,
    status: str,
    error: str | None = None,
    exit_code: int | None = None,
) -> dict[str, Any]:
    """Build summary/metrics only from already persisted panel files.

    This function deliberately does not contact the Agent API and therefore is
    safe to call after a deadline, signal, or child failure.
    """

    panels = [_read_panel(run_dir / "panels" / f"{task_id}.json", task_id) for task_id in task_ids]
    unknown = [panel for panel in panels if panel.get("status") == "unknown"]
    summary = {
        "schema_version": "t028-summary-v1",
        "mode": mode,
        "status": status,
        "error": error,
        "exit_code": exit_code,
        "task_count": len(panels),
        "terminal_count": len(panels) - len(unknown),
        "unknown_count": len(unknown),
        "results": panels,
        "runtime_evidence": collect_runtime_evidence(run_dir),
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "metrics.json", summary)
    return summary


def persist_unknown_panels(
    run_dir: Path,
    task_ids: tuple[str, ...] | list[str],
    public_tasks: dict[str, dict[str, Any]],
    *,
    reason: str,
) -> None:
    """Persist unknown only for tasks without a terminal panel."""

    for task_id in task_ids:
        panel_path = run_dir / "panels" / f"{task_id}.json"
        if panel_path.is_file():
            existing = _read_panel(panel_path, task_id)
            if existing.get("status") != "unknown":
                continue
        public = public_tasks.get(task_id, {})
        write_json(
            panel_path,
            {
                "task_id": task_id,
                "prompt": public.get("prompt"),
                "split": public.get("split"),
                "status": "unknown",
                "qualification": "technical_unknown",
                "evaluation_status": "technical_unknown",
                "terminal_reason": reason,
            },
        )


def run_smoke(
    profile: ServiceProfile,
    run_dir: Path,
    *,
    smoke_timeout_s: float,
    clock: LifecycleClock | None = None,
    signal_controller: SignalController | None = None,
) -> dict[str, Any]:
    if smoke_timeout_s <= 0:
        raise ValueError("smoke_timeout_s_must_be_positive")
    clock = clock or LifecycleClock(
        soft_cutoff_s=max(30.0, smoke_timeout_s * max(len(profile.benchmark_task_ids), 1)),
        cleanup_reserve_s=0,
    )
    task_ids = tuple(profile.benchmark_task_ids)
    public_tasks: dict[str, dict[str, Any]] = {}
    try:
        if signal_controller is not None:
            signal_controller.raise_if_requested()
        status, benchmark_payload = json_request(
            f"{profile.agent_endpoint}/benchmarks",
            timeout_s=clock.timeout(30.0, "benchmark_list"),
        )
        if status != 200:
            raise RuntimeError(f"benchmarks_http_status:{status}")
        tasks = benchmark_payload.get("tasks")
        if not isinstance(tasks, list):
            raise RuntimeError("benchmarks_payload_invalid")
        task_by_id = {item.get("task_id"): item for item in tasks if isinstance(item, dict)}
        if set(task_by_id) != set(task_ids):
            raise RuntimeError("benchmarks_public_task_set_mismatch")
        if any("reference_sql" in item or "constraints" in item for item in tasks):
            raise RuntimeError("private_benchmark_field_exposed")
        public_tasks = {task_id: task_by_id[task_id] for task_id in task_ids}
        write_json(run_dir / "http" / "benchmarks.json", benchmark_payload)

        for task_id in task_ids:
            if signal_controller is not None:
                signal_controller.raise_if_requested()
            clock.check(f"submit:{task_id}")
            public = public_tasks[task_id]
            request_id = f"{run_dir.name}-{task_id}"
            request_payload = {
                "prompt": public["prompt"],
                "benchmark_task_id": task_id,
            }
            status, submitted = json_request(
                f"{profile.agent_endpoint}/tasks",
                method="POST",
                payload=request_payload,
                headers={"X-Request-ID": request_id},
                timeout_s=clock.timeout(30.0, f"submit:{task_id}"),
            )
            write_json(run_dir / "http" / f"{task_id}.submit.json", submitted)
            if status != 202:
                raise RuntimeError(f"task_submit_http_status:{task_id}:{status}")
            job_id = submitted.get("id")
            if not isinstance(job_id, str):
                raise RuntimeError(f"task_submit_missing_job_id:{task_id}")
            task_deadline = min(clock.run_deadline, time.monotonic() + smoke_timeout_s)
            history = []
            while True:
                if signal_controller is not None:
                    signal_controller.raise_if_requested()
                remaining = min(task_deadline - time.monotonic(), clock.remaining_s)
                if remaining <= 0:
                    if clock.remaining_s <= 0:
                        raise DeadlineExceeded(f"poll:{task_id}")
                    raise TimeoutError(f"task_poll_timeout:{task_id}")
                status, current = json_request(
                    f"{profile.agent_endpoint}/tasks/{job_id}",
                    timeout_s=min(30.0, remaining),
                )
                history.append({"http_status": status, "payload": current})
                if status != 200:
                    raise RuntimeError(f"task_status_http_status:{task_id}:{status}")
                if current.get("status") not in ("queued", "running"):
                    break
                time.sleep(min(0.25, max(0.01, remaining)))
            write_json(run_dir / "http" / f"{task_id}.status.json", current)
            write_json(run_dir / "http" / f"{task_id}.status-history.json", history)
            trajectory = None
            if current.get("run_id"):
                status, trajectory = json_request(
                    f"{profile.agent_endpoint}/tasks/{job_id}/trajectory",
                    timeout_s=clock.timeout(30.0, f"trajectory:{task_id}"),
                )
                write_json(run_dir / "http" / f"{task_id}.trajectory.json", trajectory)
                if status != 200:
                    raise RuntimeError(f"trajectory_http_status:{task_id}:{status}")
            panel = {
                "task_id": task_id,
                "prompt": public["prompt"],
                "split": public.get("split"),
                "request_id": request_id,
                "job_id": job_id,
                "status": current.get("status"),
                "qualification": current.get("qualification"),
                "evaluation_status": current.get("evaluation_status"),
                "terminal_reason": current.get("terminal_reason"),
                "reward": current.get("reward"),
                "trajectory_path": current.get("trajectory_path"),
                "model_version": current.get("model_version"),
                "database_sha256": trajectory.get("database_sha256") if trajectory else None,
                "model_attempts": len(trajectory.get("model_attempts", [])) if trajectory else None,
                "tool_calls": sum(
                    1
                    for step in (trajectory or {}).get("steps", [])
                    if step.get("tool_name")
                ),
                "prompt_tokens": sum(
                    step.get("prompt_tokens", 0) for step in (trajectory or {}).get("steps", [])
                )
                if trajectory
                else None,
                "completion_tokens": sum(
                    step.get("completion_tokens", 0) for step in (trajectory or {}).get("steps", [])
                )
                if trajectory
                else None,
            }
            write_json(run_dir / "panels" / f"{task_id}.json", panel)
            finalize_results(
                run_dir,
                task_ids,
                mode="smoke",
                status="RUNNING",
                error=None,
            )
        return finalize_results(run_dir, task_ids, mode="smoke", status="COMPLETED")
    except Exception as exc:
        reason = f"{type(exc).__name__}:{exc}"
        persist_unknown_panels(run_dir, task_ids, public_tasks, reason=reason)
        finalize_results(run_dir, task_ids, mode="smoke", status="INTERRUPTED", error=reason)
        raise


def wait_for_serve(
    processes: dict[str, subprocess.Popen],
    *,
    clock: LifecycleClock,
    signal_controller: SignalController | None = None,
) -> None:
    """Keep ready services alive without scheduling benchmark work."""

    while True:
        if signal_controller is not None:
            signal_controller.raise_if_requested()
        for name, process in processes.items():
            if process.poll() is not None:
                raise ServiceProcessExited(name, process.returncode)
        remaining = clock.remaining_s
        if remaining <= 0:
            raise DeadlineExceeded("serve_wait")
        time.sleep(min(0.25, max(0.01, remaining)))


def write_private_verifier_inputs(profile: ServiceProfile, run_dir: Path) -> None:
    registry = profile.load_benchmark_registry()
    for task_id, task in registry.items():
        write_json(
            run_dir / "verifier-private" / f"{task_id}.json",
            {
                "task_id": task_id,
                "reference_sql": task.reference_sql,
                "ordered": task.ordered,
                "min_sql_calls": task.min_sql_calls,
                "required_probe_sql": task.required_probe_sql,
                "required_schema": task.required_schema,
                "required_sql_error": task.required_sql_error,
            },
        )


def write_manifest(run_dir: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        entries.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {"schema_version": "t027-manifest-v1", "entries": entries}
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def write_report(
    run_dir: Path,
    profile: ServiceProfile,
    *,
    mode: str,
    supervisor_status: str,
    exit_code: int,
    process_records: list[dict[str, Any]],
    error: str | None,
) -> None:
    lines = [
        "# T-027 bounded SFT N service lifecycle",
        "",
        f"- mode: `{mode}`",
        f"- supervisor status: `{supervisor_status}`",
        f"- supervisor exit code: `{exit_code}`",
        f"- profile: `{profile.profile_id}` / model version `{profile.model_version}`",
        f"- model: `{profile.model_id}@{profile.revision}`",
        f"- adapter SHA-256: `{profile.adapter_sha256}`",
        "- runtime Settings: `"
        + json.dumps(profile.runtime.model_dump(), ensure_ascii=False, sort_keys=True)
        + "`",
        "- generation: `"
        + json.dumps(profile.generation.model_dump(), ensure_ascii=False, sort_keys=True)
        + "`",
        f"- run directory: `{run_dir}`",
        "",
        "The run is a local, single-process-per-service loopback lifecycle check. "
        "`smoke` schedules only the fixed public train tasks; `serve` waits for "
        "external clients and never schedules a benchmark; `startup-check` is the "
        "backward-compatible `--no-smoke` mode. This is not an online deployment "
        "or a generalization evaluation. The prior "
        "frozen E-026 panel reported SFT N 8/20 (40%) versus Base 0/20; this run "
        "does not change that denominator or claim online generalization.",
        "",
        "## Process and technical qualification",
        "",
    ]
    for record in process_records:
        lines.append(
            f"- {record['name']}: pid `{record['pid']}`, return code `{record.get('returncode')}`, "
            f"owned process group `{record['process_group_owned']}`; stdout `{record['stdout']}`, "
            f"stderr `{record['stderr']}`."
        )
    if error:
        lines.extend(["", f"- supervisor error: `{error}`"])
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        lines.extend(["", "## Persisted results", ""])
        for result in summary.get("results", []):
            reward = result.get("reward") or {}
            lines.append(
                f"- `{result['task_id']}`: qualification=`{result.get('qualification')}`, "
                f"evaluation=`{result.get('evaluation_status')}`, "
                f"business_success=`{reward.get('success') if reward else None}`, "
                f"terminal=`{result.get('terminal_reason')}`, "
                f"model_attempts=`{result.get('model_attempts')}`, "
                f"tool_calls=`{result.get('tool_calls')}`."
            )
        lines.append(
            f"- terminal panels: `{summary.get('terminal_count')}`, "
            f"unknown panels: `{summary.get('unknown_count')}`."
        )
        lines.extend(["", "## Runtime evidence", ""])
        for name, evidence in summary.get("runtime_evidence", {}).items():
            lines.append(
                f"- `{name}`: `{evidence.get('value')}`; source="
                f"`{evidence.get('source')}`, definition=`{evidence.get('definition')}`, "
                f"reason=`{evidence.get('reason')}`."
            )
    lines.extend(
        [
            "",
            "Private verifier SQL and constraints are stored only under "
            "`verifier-private/`; the public panels and `/benchmarks` response contain "
            "task_id, prompt and split only. Model-generated requests/responses and "
            "event IDs remain in the model event log and trajectory evidence.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=Path("configs/serving/sft-n.toml"))
    parser.add_argument(
        "--python",
        "--python-executable",
        dest="python_executable",
        type=Path,
        help="venv entry point; symlink spelling is preserved in child argv",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model-port", type=int)
    parser.add_argument("--agent-port", type=int)
    parser.add_argument("--auto-ports", action="store_true")
    parser.add_argument("--mode", choices=("smoke", "serve"), default="smoke")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=300)
    parser.add_argument("--smoke-timeout", type=float, default=720)
    parser.add_argument("--soft-cutoff", type=float, default=1320)
    parser.add_argument("--cleanup-reserve", type=float, default=60)
    parser.add_argument("--cleanup-grace", type=float, default=15)
    return parser.parse_args()




def _bounded_main() -> int:
    """Run the T-027 supervisor with one deadline and durable partial results."""

    supervisor_started = time.monotonic()
    args = parse_args()
    if args.no_smoke and args.mode != "smoke":
        raise SystemExit("--no-smoke is only compatible with the default smoke mode")
    if args.ready_timeout <= 0 or args.smoke_timeout <= 0:
        raise SystemExit("--ready-timeout and --smoke-timeout must be positive")
    mode = "startup-check" if args.no_smoke else args.mode
    clock = LifecycleClock(
        soft_cutoff_s=args.soft_cutoff,
        cleanup_reserve_s=args.cleanup_reserve,
        started_at=supervisor_started,
    )
    signal_controller = SignalController()
    global _ACTIVE_SIGNAL_CONTROLLER
    _ACTIVE_SIGNAL_CONTROLLER = signal_controller
    previous_signal_handlers = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }
    signal.signal(signal.SIGTERM, handle_supervisor_signal)
    signal.signal(signal.SIGINT, handle_supervisor_signal)

    root = args.repo_root.resolve()
    profile: ServiceProfile | None = None
    run_dir: Path | None = None
    commands: dict[str, Any] = {}
    processes: dict[str, subprocess.Popen] = {}
    process_records: list[dict[str, Any]] = []
    supervisor_status = "NOT_STARTED"
    error: str | None = None
    exit_code = 1

    try:
        profile_path = absolute_path_without_resolving(args.profile, base=root)
        profile = ServiceProfile.load(profile_path, repo_root=root)
        clock.check("profile_load")
        requested_run_dir = (
            absolute_path_without_resolving(args.run_dir, base=root)
            if args.run_dir is not None
            else None
        )
        run_dir = allocate_run_dir(
            profile,
            requested_run_dir,
            default_root=root / DEFAULT_RUN_ROOT,
        )
        clock.check("run_directory_setup")
        source_preflight = profile.preflight()
        clock.check("preflight")
        database, database_copy = copy_read_only_database(profile, run_dir)
        clock.check("database_setup")

        preferred_model_port = args.model_port or int(
            urlparse(profile.model_completion_endpoint).port or 8001
        )
        preferred_agent_port = args.agent_port or int(urlparse(profile.agent_endpoint).port or 8000)
        model_port = choose_port(preferred_model_port, args.auto_ports)
        agent_port = choose_port(preferred_agent_port, args.auto_ports)
        if model_port == agent_port:
            agent_port = probe_free_loopback_port()
        clock.check("port_setup")
        profile = profile.with_paths(database=database, output=run_dir).with_endpoints(
            model_completion=f"http://127.0.0.1:{model_port}/v1/chat/completions",
            model_health=f"http://127.0.0.1:{model_port}/health",
            agent=f"http://127.0.0.1:{agent_port}",
        )

        write_json(run_dir / "service-profile.json", profile.model_dump(mode="json"))
        write_json(run_dir / "asset-preflight.json", source_preflight)
        write_json(
            run_dir / "input-manifest.json",
            {
                "schema_version": "t027-input-manifest-v1",
                "profile_id": profile.profile_id,
                "model_version": profile.model_version,
                "model_id": profile.model_id,
                "revision": profile.revision,
                "adapter_sha256": profile.adapter_sha256,
                "adapter_config_sha256": profile.adapter_config_sha256,
                "benchmark_records_sha256": profile.benchmark_records_sha256,
                "benchmark_task_ids": list(profile.benchmark_task_ids),
                "database": database_copy,
                "source_database_physical_sha256": profile.database_physical_sha256,
                "source_database_content_sha256": profile.database_content_sha256,
                "model_endpoint": profile.model_completion_endpoint,
                "model_health_endpoint": profile.model_health_endpoint,
                "agent_endpoint": profile.agent_endpoint,
            },
        )
        write_private_verifier_inputs(profile, run_dir)
        clock.check("input_setup")

        python_executable = resolve_python_executable(args.python_executable, root=root)
        if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
            raise FileNotFoundError(f"gpu_python_missing:{python_executable}")
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(root / "src")
                + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
        model_command = [
            str(python_executable),
            "-m",
            "enterprise_agent_rl.model_server",
            "--profile",
            str(profile_path),
            "--repo-root",
            str(root),
            "--output",
            str(run_dir / "model"),
            "--port",
            str(model_port),
            "--model-endpoint",
            profile.model_completion_endpoint,
            "--health-endpoint",
            profile.model_health_endpoint,
        ]
        api_command = [
            str(python_executable),
            "-m",
            "enterprise_agent_rl.cli",
            "serve",
            "--profile",
            str(profile_path),
            "--repo-root",
            str(root),
            "--database",
            str(database),
            "--output",
            str(run_dir / "api"),
            "--port",
            str(agent_port),
            "--model-endpoint",
            profile.model_completion_endpoint,
            "--model-health-endpoint",
            profile.model_health_endpoint,
        ]
        commands = {
            "schema_version": "t027-commands-v1",
            "repo_root": str(root),
            "python": str(python_executable),
            "python_entrypoint_preserved": str(python_executable),
            "profile": str(profile_path),
            "model": model_command,
            "agent_api": api_command,
            "mode": mode,
            "legacy_no_smoke": args.no_smoke,
            "model_port": model_port,
            "agent_port": agent_port,
            "ready_timeout_seconds": args.ready_timeout,
            "smoke_timeout_seconds": args.smoke_timeout,
            "soft_cutoff_seconds": args.soft_cutoff,
            "cleanup_reserve_seconds": args.cleanup_reserve,
            "cleanup_grace_seconds": args.cleanup_grace,
        }
        write_json(run_dir / "commands.json", commands)
        clock.check("command_setup")

        model_process, model_record = spawn_service(
            "model", model_command, root=root, run_dir=run_dir, env=env
        )
        processes["model"] = model_process
        process_records.append(model_record)
        clock.check("model_spawn")
        model_health = wait_for_ready(
            profile.model_health_endpoint,
            model_process,
            min(profile.model_load_timeout_s, args.ready_timeout),
            "model",
            clock=clock,
            signal_controller=signal_controller,
        )
        write_json(run_dir / "health" / "model-ready.json", model_health)
        write_service_snapshot(
            run_dir,
            "model-health",
            profile.model_health_endpoint,
            clock=clock,
            signal_controller=signal_controller,
        )
        write_service_snapshot(
            run_dir,
            "model-models",
            f"http://127.0.0.1:{model_port}/models",
            clock=clock,
            signal_controller=signal_controller,
        )

        clock.check("agent_spawn")
        api_process, api_record = spawn_service(
            "agent-api", api_command, root=root, run_dir=run_dir, env=env
        )
        processes["agent-api"] = api_process
        process_records.append(api_record)
        api_health = wait_for_ready(
            f"{profile.agent_endpoint}/health",
            api_process,
            args.ready_timeout,
            "agent_api",
            clock=clock,
            signal_controller=signal_controller,
        )
        write_json(run_dir / "health" / "agent-ready.json", api_health)
        write_service_snapshot(
            run_dir,
            "agent-health",
            f"{profile.agent_endpoint}/health",
            clock=clock,
            signal_controller=signal_controller,
        )
        write_service_snapshot(
            run_dir,
            "agent-models",
            f"{profile.agent_endpoint}/models",
            clock=clock,
            signal_controller=signal_controller,
        )
        clock.check("ready")
        if mode == "smoke":
            run_smoke(
                profile,
                run_dir,
                smoke_timeout_s=args.smoke_timeout,
                clock=clock,
                signal_controller=signal_controller,
            )
            for name, process in processes.items():
                if process.poll() is not None:
                    raise ServiceProcessExited(name, process.returncode)
            supervisor_status = "COMPLETED"
            exit_code = 0
        elif mode == "startup-check":
            finalize_results(run_dir, [], mode=mode, status="STARTUP_CHECK_COMPLETED", exit_code=0)
            supervisor_status = "STARTUP_CHECK_COMPLETED"
            exit_code = 0
        else:
            finalize_results(run_dir, [], mode=mode, status="SERVING", exit_code=None)
            wait_for_serve(processes, clock=clock, signal_controller=signal_controller)
            supervisor_status = "COMPLETED"
            exit_code = 0
    except SupervisorSignal as exc:
        error = f"{type(exc).__name__}:{exc}"
        supervisor_status = "STOPPED_BY_SIGNAL"
        exit_code = exc.exit_code
    except DeadlineExceeded as exc:
        error = f"{type(exc).__name__}:{exc}"
        supervisor_status = "DEADLINE_EXCEEDED"
        exit_code = 1
    except ServiceProcessExited as exc:
        error = f"{type(exc).__name__}:{exc}"
        supervisor_status = "CHILD_PROCESS_FAILURE"
        exit_code = 1
    except TimeoutError as exc:
        error = f"{type(exc).__name__}:{exc}"
        supervisor_status = "DEADLINE_EXCEEDED" if clock.remaining_s <= 0 else "FAILED"
        exit_code = 1
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        supervisor_status = "FAILED"
        exit_code = 1
    finally:
        signal_controller.cleaning = True
        for name in reversed(("agent-api", "model")):
            termination: dict[str, Any] = {}
            returncode = stop_process_group(
                processes.get(name),
                grace_s=args.cleanup_grace,
                cleanup_deadline=clock.cleanup_deadline,
                evidence=termination,
            )
            for record in process_records:
                if record["name"] == name:
                    record["returncode"] = returncode
                    record["termination"] = termination
        commands["process_returncodes"] = {
            record["name"]: record.get("returncode") for record in process_records
        }
        commands["supervisor_status"] = supervisor_status
        commands["supervisor_error"] = error
        commands["exit_code"] = exit_code
        if run_dir is not None:
            write_json(run_dir / "commands.json", commands)
            if profile is not None:
                task_ids = tuple(profile.benchmark_task_ids) if mode == "smoke" else []
                if mode == "smoke":
                    persist_unknown_panels(
                        run_dir,
                        task_ids,
                        {},
                        reason=error or "supervisor_not_completed",
                    )
                final_summary = finalize_results(
                    run_dir,
                    task_ids,
                    mode=mode,
                    status=supervisor_status,
                    error=error,
                    exit_code=exit_code,
                )
                runtime_evidence = final_summary["runtime_evidence"]
                slurm_job_id = os.environ.get("SLURM_JOB_ID")
                write_json(
                    run_dir / "job-chain.json",
                    {
                        "schema_version": "t027-job-chain-v1",
                        "task": "T-027",
                        "experiment": None,
                        "slurm_job_id": slurm_job_id,
                        "partition": os.environ.get("SLURM_JOB_PARTITION"),
                        "job_chain": [slurm_job_id] if slurm_job_id else [],
                        "status": supervisor_status,
                        "exit_code": exit_code,
                        "error": error,
                    },
                )
                write_json(
                    run_dir / "allocation-ledger.json",
                    {
                        "schema_version": "t027-allocation-ledger-v1",
                        "task": "T-027",
                        "experiment": None,
                        "supervisor_wall_clock": {
                            "elapsed_seconds": clock.elapsed_s,
                            "soft_cutoff_seconds": clock.soft_cutoff_s,
                            "cleanup_reserve_seconds": clock.cleanup_reserve_s,
                            "deadline_enforced": True,
                        },
                        "slurm_allocation": {
                            "job_id": slurm_job_id,
                            "partition": os.environ.get("SLURM_JOB_PARTITION"),
                            "time_limit_raw": os.environ.get("SLURM_JOB_TIMELIMIT"),
                            "observed_allocation_seconds": None,
                            "source": "scheduler environment only; no scheduler query by T-027",
                        },
                        "cross_job_budget": {
                            "automatic_guard_implemented": False,
                            "historical_allocations": [],
                            "remaining_total_budget_seconds": None,
                            "note": (
                                "Cross-job cumulative accounting requires the executor's "
                                "explicit job chain; unknown values are not filled with zero."
                            ),
                        },
                    },
                )
                write_json(
                    run_dir / "checks.json",
                    {
                        "schema_version": "t028-checks-v1",
                        "mode": mode,
                        "supervisor_status": supervisor_status,
                        "exit_code": exit_code,
                        "job_id": slurm_job_id,
                        "gpu_calls": None,
                        "model_calls": runtime_evidence["model_calls"]["value"],
                        "business_sql_calls": runtime_evidence["business_sql_calls"]["value"],
                        "verifier_calls": runtime_evidence["verifier_calls"]["value"],
                        "runtime_evidence": runtime_evidence,
                        "gpu_allocation": {
                            "allocation_seconds": None,
                            "allocation_source": (
                                "allocation-ledger.slurm_allocation."
                                "observed_allocation_seconds"
                            ),
                            "allocation_reason": (
                                "scheduler allocation is not queried by the launcher"
                            ),
                            "wall_clock_seconds": clock.elapsed_s,
                            "wall_clock_source": "LifecycleClock.elapsed_s",
                        },
                        "count_compatibility": {
                            "gpu_calls": (
                                "null by design: no per-run GPU event counter; use gpu_allocation "
                                "and allocation-ledger instead"
                            ),
                            "other_counts": (
                                "counts are values from runtime_evidence; missing artifacts "
                                "remain null "
                                "and are not inferred as zero"
                            ),
                        },
                        "manifest_excludes_self": True,
                    },
                )
                write_json(run_dir / "process" / "records.json", {"processes": process_records})
                write_report(
                    run_dir,
                    profile,
                    mode=mode,
                    supervisor_status=supervisor_status,
                    exit_code=exit_code,
                    process_records=process_records,
                    error=error,
                )
                write_manifest(run_dir)
                (run_dir / "manifest.sha256").write_text(
                    sha256_file(run_dir / "manifest.json") + "  manifest.json\n"
                )
        signal.signal(signal.SIGTERM, previous_signal_handlers[signal.SIGTERM])
        signal.signal(signal.SIGINT, previous_signal_handlers[signal.SIGINT])
        _ACTIVE_SIGNAL_CONTROLLER = None
    return exit_code


main = _bounded_main


if __name__ == "__main__":
    raise SystemExit(main())
