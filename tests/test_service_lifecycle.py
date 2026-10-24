"""CPU-only lifecycle contracts using real owned subprocesses and loopback HTTP."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from enterprise_agent_rl.service_lifecycle import (
    DeadlineExceeded,
    LifecycleClock,
    ServiceProcessExited,
)

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts/serving/run_sft_n_api.py"
SPEC = importlib.util.spec_from_file_location("t027_launcher", LAUNCHER_PATH)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


FAKE_SERVER = textwrap.dedent(
    r'''
    import json
    import os
    import sys
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    port = int(sys.argv[1])
    kind = sys.argv[2]
    mode = os.environ.get("T027_FAKE_MODE", "normal")
    ready_delay = float(os.environ.get("T027_READY_DELAY", "0"))
    jobs = {}
    counter = 0
    tasks = [
        {"task_id": "task-one", "prompt": "public one", "split": "train"},
        {"task_id": "task-two", "prompt": "public two", "split": "train"},
    ]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def send_json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - stdlib handler API
            if self.path == "/health":
                if kind == "model" and ready_delay:
                    time.sleep(ready_delay)
                self.send_json(200, {"status": "ready" if kind == "model" else "ok"})
            elif self.path == "/models":
                self.send_json(200, {"models": [{"version": "fake-cpu"}]})
            elif kind == "api" and self.path == "/benchmarks":
                self.send_json(200, {"tasks": tasks})
            elif kind == "api" and self.path.startswith("/tasks/"):
                job_id = self.path.rsplit("/", 1)[-1]
                if job_id not in jobs:
                    self.send_json(404, {"error": "unknown_task"})
                    return
                status = "queued" if mode == "poll_slow" else "completed"
                self.send_json(
                    200,
                    {
                        "id": job_id,
                        "status": status,
                        "qualification": "qualified",
                        "evaluation_status": "verified_benchmark",
                        "terminal_reason": "completed" if status == "completed" else None,
                        "reward": {"success": True} if status == "completed" else None,
                    },
                )
            else:
                self.send_json(404, {"error": "not_found"})

        def do_POST(self):  # noqa: N802 - stdlib handler API
            global counter
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if kind != "api" or self.path != "/tasks":
                self.send_json(404, {"error": "not_found"})
                return
            if mode == "second_submit_failure" and payload.get("benchmark_task_id") == "task-two":
                self.send_json(503, {"error": "fixture_second_task_failure"})
                return
            counter += 1
            job_id = "job-" + str(counter)
            jobs[job_id] = payload
            self.send_json(202, {"id": job_id, "status": "queued"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.serve_forever()
    '''
)


def fake_profile(model_port: int, api_port: int):
    return SimpleNamespace(
        model_health_endpoint=f"http://127.0.0.1:{model_port}/health",
        agent_endpoint=f"http://127.0.0.1:{api_port}",
        benchmark_task_ids=("task-one", "task-two"),
    )


class FakeDump:
    def model_dump(self):
        return {}


class FakeSupervisorProfile:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.database_path = output_dir / "source.sqlite3"
        self.model_completion_endpoint = "http://127.0.0.1:8001/v1/chat/completions"
        self.model_health_endpoint = "http://127.0.0.1:8001/health"
        self.agent_endpoint = "http://127.0.0.1:8000"
        self.profile_id = "fake-cpu-profile"
        self.model_version = "fake-cpu"
        self.model_id = "fake-model"
        self.revision = "fake-revision"
        self.adapter_sha256 = "adapter"
        self.adapter_config_sha256 = "adapter-config"
        self.benchmark_records_sha256 = "records"
        self.database_physical_sha256 = "database-physical"
        self.database_content_sha256 = "database-content"
        self.benchmark_task_ids = ("task-one", "task-two")
        self.model_load_timeout_s = 2
        self.health_timeout_s = 1
        self.runtime = FakeDump()
        self.generation = FakeDump()

    def with_paths(self, *, database=None, output=None):
        result = copy.copy(self)
        if database is not None:
            result.database_path = database
        if output is not None:
            result.output_dir = output
        return result

    def with_endpoints(self, *, model_completion=None, model_health=None, agent=None):
        result = copy.copy(self)
        if model_completion is not None:
            result.model_completion_endpoint = model_completion
        if model_health is not None:
            result.model_health_endpoint = model_health
        if agent is not None:
            result.agent_endpoint = agent
        return result

    def preflight(self):
        return {"fake_fixture": True}

    def model_dump(self, mode="python"):
        return {
            "profile_id": self.profile_id,
            "model_version": self.model_version,
            "model_completion_endpoint": self.model_completion_endpoint,
            "model_health_endpoint": self.model_health_endpoint,
            "agent_endpoint": self.agent_endpoint,
        }


def start_fake_pair(tmp_path: Path, *, mode: str = "normal", ready_delay: float = 0):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_port = launcher.probe_free_loopback_port()
    api_port = launcher.probe_free_loopback_port()
    env = os.environ.copy()
    env["T027_FAKE_MODE"] = mode
    env["T027_READY_DELAY"] = str(ready_delay)
    processes = {}
    records = []
    for name, port, kind in (
        ("model", model_port, "model"),
        ("agent-api", api_port, "api"),
    ):
        process, record = launcher.spawn_service(
            name,
            [sys.executable, "-c", FAKE_SERVER, str(port), kind],
            root=ROOT,
            run_dir=run_dir,
            env=env,
        )
        processes[name] = process
        records.append(record)
    return run_dir, fake_profile(model_port, api_port), processes, records


def stop_pair(processes):
    for name in ("agent-api", "model"):
        process = processes.get(name)
        if process is not None:
            launcher.stop_process_group(process, grace_s=0.2)


def wait_pair(profile, processes, clock):
    launcher.wait_for_ready(
        profile.model_health_endpoint,
        processes["model"],
        2,
        "model",
        clock=clock,
    )
    launcher.wait_for_ready(
        f"{profile.agent_endpoint}/health",
        processes["agent-api"],
        2,
        "agent_api",
        clock=clock,
    )


def test_smoke_persists_each_terminal_result_and_offline_finalize(tmp_path):
    run_dir, profile, processes, _records = start_fake_pair(tmp_path)
    try:
        clock = LifecycleClock(soft_cutoff_s=5, cleanup_reserve_s=2)
        wait_pair(profile, processes, clock)
        result = launcher.run_smoke(profile, run_dir, smoke_timeout_s=2, clock=clock)
        assert result["status"] == "COMPLETED"
        assert result["terminal_count"] == 2
        assert all(
            json.loads((run_dir / "panels" / f"{task_id}.json").read_text())["status"]
            == "completed"
            for task_id in profile.benchmark_task_ids
        )
        before = (run_dir / "summary.json").read_text()
        finalized = launcher.finalize_results(
            run_dir, profile.benchmark_task_ids, mode="smoke", status="OFFLINE_FINALIZED"
        )
        assert finalized["terminal_count"] == 2
        assert (run_dir / "summary.json").read_text() != before
        assert not (run_dir / "panels" / "task-one.json").read_text().find("reference_sql") >= 0
    finally:
        stop_pair(processes)


def test_launcher_main_smoke_cli_uses_production_supervisor_path(tmp_path, monkeypatch):
    profile = FakeSupervisorProfile(tmp_path / "profile-output")
    profile.output_dir.mkdir()
    real_spawn = launcher.spawn_service

    def fake_spawn(name, _command, *, root, run_dir, env):
        port = int(_command[_command.index("--port") + 1])
        kind = "model" if name == "model" else "api"
        return real_spawn(
            name,
            [sys.executable, "-c", FAKE_SERVER, str(port), kind],
            root=root,
            run_dir=run_dir,
            env=env,
        )

    def fake_copy(_profile, run_dir):
        database = run_dir / "inputs" / "fake.sqlite3"
        database.parent.mkdir(parents=True, exist_ok=True)
        database.write_text("cpu-fixture\n")
        return database, {"copy": str(database), "physical_sha256": "fixture"}

    monkeypatch.setattr(
        launcher.ServiceProfile,
        "load",
        staticmethod(lambda _path, repo_root=None: profile),
    )
    monkeypatch.setattr(launcher, "copy_read_only_database", fake_copy)
    monkeypatch.setattr(launcher, "write_private_verifier_inputs", lambda *_args: None)
    monkeypatch.setattr(launcher, "spawn_service", fake_spawn)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sft_n_api.py",
            "--repo-root",
            str(ROOT),
            "--profile",
            "fake-profile.toml",
            "--run-dir",
            str(tmp_path / "cli-run"),
            "--python-executable",
            sys.executable,
            "--auto-ports",
            "--ready-timeout",
            "2",
            "--smoke-timeout",
            "2",
            "--soft-cutoff",
            "5",
            "--cleanup-reserve",
            "1",
        ],
    )
    assert launcher.main() == 0
    run_dir = tmp_path / "cli-run"
    commands = json.loads((run_dir / "commands.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert commands["mode"] == "smoke"
    assert commands["python_entrypoint_preserved"] == sys.executable
    assert summary["status"] == "COMPLETED"
    assert summary["terminal_count"] == 2


def test_main_default_allocates_unique_runs_and_rejects_existing_explicit_path(
    tmp_path, monkeypatch
):
    repo_root = tmp_path / "repo-root"
    repo_root.mkdir()
    default_root = repo_root / "artifacts" / "service-runs"
    prior_run = default_root / "prior-run"
    prior_run.mkdir(parents=True)
    sentinel = prior_run / "sentinel.txt"
    sentinel.write_bytes(b"keep-this-run\n")
    sentinel_before = sentinel.read_bytes()

    profile = FakeSupervisorProfile(tmp_path / "profile-output")
    profile.output_dir.mkdir()
    real_spawn = launcher.spawn_service

    def fake_spawn(name, _command, *, root, run_dir, env):
        port = int(_command[_command.index("--port") + 1])
        kind = "model" if name == "model" else "api"
        return real_spawn(
            name,
            [sys.executable, "-c", FAKE_SERVER, str(port), kind],
            root=root,
            run_dir=run_dir,
            env=env,
        )

    def fake_copy(_profile, run_dir):
        database = run_dir / "inputs" / "fake.sqlite3"
        database.parent.mkdir(parents=True, exist_ok=True)
        database.write_text("cpu-fixture\n")
        return database, {"copy": str(database), "physical_sha256": "fixture"}

    monkeypatch.setattr(
        launcher.ServiceProfile,
        "load",
        staticmethod(lambda _path, repo_root=None: profile),
    )
    monkeypatch.setattr(launcher, "copy_read_only_database", fake_copy)
    monkeypatch.setattr(launcher, "write_private_verifier_inputs", lambda *_args: None)
    monkeypatch.setattr(launcher, "spawn_service", fake_spawn)

    def invoke(extra: list[str] | None = None) -> int:
        argv = [
            "run_sft_n_api.py",
            "--repo-root",
            str(repo_root),
            "--profile",
            "fake-profile.toml",
            "--python-executable",
            sys.executable,
            "--auto-ports",
            "--ready-timeout",
            "2",
            "--smoke-timeout",
            "2",
            "--soft-cutoff",
            "5",
            "--cleanup-reserve",
            "1",
        ]
        if extra:
            argv.extend(extra)
        monkeypatch.setattr(sys, "argv", argv)
        return launcher.main()

    assert invoke() == 0
    assert invoke() == 0
    run_dirs = sorted(
        path
        for path in default_root.iterdir()
        if path.is_dir() and path.name != "prior-run"
    )
    assert len(run_dirs) == 2
    assert run_dirs[0] != run_dirs[1]
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        checks = json.loads((run_dir / "checks.json").read_text())
        assert summary["status"] == "COMPLETED"
        assert checks["gpu_calls"] is None
        assert checks["model_calls"] is None
        assert checks["business_sql_calls"] is None
        assert checks["verifier_calls"] is None
        assert checks["runtime_evidence"]["model_calls"]["reason"] is not None

    assert invoke(["--run-dir", str(prior_run)]) == 1
    assert sentinel.read_bytes() == sentinel_before


def test_second_task_failure_keeps_first_panel_and_marks_unknown(tmp_path):
    run_dir, profile, processes, _records = start_fake_pair(
        tmp_path, mode="second_submit_failure"
    )
    try:
        clock = LifecycleClock(soft_cutoff_s=5, cleanup_reserve_s=2)
        wait_pair(profile, processes, clock)
        with pytest.raises(RuntimeError, match="task_submit_http_status:task-two:503"):
            launcher.run_smoke(profile, run_dir, smoke_timeout_s=2, clock=clock)
        first = json.loads((run_dir / "panels/task-one.json").read_text())
        second = json.loads((run_dir / "panels/task-two.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        assert first["status"] == "completed"
        assert second["status"] == "unknown"
        assert second["evaluation_status"] == "technical_unknown"
        assert summary["terminal_count"] == 1
        assert summary["unknown_count"] == 1
    finally:
        stop_pair(processes)


def test_global_deadline_interrupts_real_slow_ready_subprocess(tmp_path):
    run_dir, profile, processes, _records = start_fake_pair(tmp_path, ready_delay=2)
    started = time.monotonic()
    try:
        clock = LifecycleClock(soft_cutoff_s=0.35, cleanup_reserve_s=1)
        with pytest.raises(DeadlineExceeded, match="model_ready"):
            launcher.wait_for_ready(
                profile.model_health_endpoint,
                processes["model"],
                5,
                "model",
                clock=clock,
            )
        assert time.monotonic() - started < 1.5
    finally:
        stop_pair(processes)


def test_global_deadline_interrupts_real_slow_poll_and_retains_unknown(tmp_path):
    run_dir, profile, processes, _records = start_fake_pair(tmp_path, mode="poll_slow")
    try:
        clock = LifecycleClock(soft_cutoff_s=0.7, cleanup_reserve_s=1)
        wait_pair(profile, processes, clock)
        with pytest.raises(DeadlineExceeded, match="poll:task-one"):
            launcher.run_smoke(profile, run_dir, smoke_timeout_s=5, clock=clock)
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["unknown_count"] == 2
        assert all(
            json.loads((run_dir / "panels" / f"{task_id}.json").read_text())["status"]
            == "unknown"
            for task_id in profile.benchmark_task_ids
        )
    finally:
        stop_pair(processes)


def test_serve_waits_for_client_without_scheduling_benchmark_then_deadlines(tmp_path):
    run_dir, profile, processes, _records = start_fake_pair(tmp_path)
    client_result = {}
    try:
        clock = LifecycleClock(soft_cutoff_s=2, cleanup_reserve_s=1)
        wait_pair(profile, processes, clock)

        def client():
            time.sleep(0.1)
            status, _payload = launcher.json_request(
                f"{profile.agent_endpoint}/health", timeout_s=1
            )
            client_result["status"] = status

        thread = threading.Thread(target=client)
        thread.start()
        with pytest.raises(DeadlineExceeded, match="serve_wait"):
            launcher.wait_for_serve(processes, clock=clock)
        thread.join(timeout=2)
        assert client_result["status"] == 200
        assert not (run_dir / "http" / "benchmarks.json").exists()
    finally:
        stop_pair(processes)


def test_child_early_exit_is_a_child_failure(tmp_path):
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    clock = LifecycleClock(soft_cutoff_s=2, cleanup_reserve_s=1)
    time.sleep(0.05)
    with pytest.raises(ServiceProcessExited, match="model_exited"):
        launcher.wait_for_ready(
            "http://127.0.0.1:1/health", process, 1, "model", clock=clock
        )


@pytest.mark.parametrize(("signum", "expected_exit"), [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_real_signal_is_forwarded_to_owned_group_and_kill_is_bounded(
    tmp_path, signum, expected_exit
):
    child_code = (
        "import signal,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True:\n"
        "    time.sleep(1)"
    )
    script = textwrap.dedent(
        f'''
        import importlib.util, json, signal, subprocess, sys, time
        from pathlib import Path
        path = Path({str(LAUNCHER_PATH)!r})
        spec = importlib.util.spec_from_file_location("t027_signal_launcher", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        child = subprocess.Popen([sys.executable, "-c", {child_code!r}], start_new_session=True)
        def handler(signum, _frame):
            raise module.SupervisorSignal(signum)
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
        time.sleep(0.25)
        print("READY", flush=True)
        try:
            while True:
                time.sleep(0.1)
        except module.SupervisorSignal as exc:
            evidence = {{}}
            returncode = module.stop_process_group(child, grace_s=0.1, evidence=evidence)
            print(json.dumps({{"returncode": returncode, "evidence": evidence}}), flush=True)
            raise SystemExit(exc.exit_code)
        '''
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == expected_exit, stderr
        payload = json.loads(stdout.strip().splitlines()[-1])
        assert payload["evidence"]["term_sent"] is True, (payload, stderr)
        assert payload["evidence"]["kill_sent"] is True, (payload, stderr)
        assert payload["evidence"]["wait_completed"] is True
        assert payload["returncode"] == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)


def test_explicit_venv_entrypoint_is_not_resolved_and_has_correct_prefix():
    requested = ROOT / ".venv-gpu/bin/python"
    entrypoint = launcher.resolve_python_executable(requested, root=ROOT)
    assert entrypoint == requested
    result = subprocess.run(
        [str(entrypoint), "-c", "import sys; print(sys.prefix)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(result.stdout.strip()) == ROOT / ".venv-gpu"
