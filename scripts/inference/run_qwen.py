"""Own the model process group and preserve evidence on every exit path."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from enterprise_agent_rl.agent import LocalModelPolicy, run_episode
from enterprise_agent_rl.base_benchmark import run_base_benchmark, write_incomplete_report
from enterprise_agent_rl.base_evaluation import evaluate_suite
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import benchmark_tasks, initialize_database
from enterprise_agent_rl.evaluation import evaluate
from enterprise_agent_rl.gpu_guard import require_compute_node
from enterprise_agent_rl.model_server import ModelSettings
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.trajectories import TrajectoryStore


def stop_model_process(process):
    """Terminate only the process group created by this supervisor."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run_protocol_smoke(policy: LocalModelPolicy, settings: Settings, output: Path) -> dict:
    """Exercise real model JSON parsing and at least one real environment tool call."""
    output.mkdir(parents=True, exist_ok=True)
    database = initialize_database(output / "protocol.sqlite3")
    task = next(t for t in benchmark_tasks() if t.public.task_id == "top-2025")
    trajectory = run_episode(task.public, policy, database, settings, output / "sql-audit.jsonl")
    store = TrajectoryStore(output / "trajectories")
    path = store.save(trajectory)
    verification = None
    verification_error = None
    try:
        reward = verify(task, trajectory, database)
        trajectory = trajectory.model_copy(update={"reward": reward})
        path = store.save(trajectory)
        verification = reward.model_dump()
    except Exception as exc:
        verification_error = {"type": type(exc).__name__, "message": str(exc)}
    outputs = [a for a in trajectory.model_attempts if a.get("raw_output") is not None]
    valid_outputs = [a for a in outputs if a.get("format_valid")]
    tool_steps = [step for step in trajectory.steps if step.tool_name]
    report = {
        "schema_version": "1.0",
        "acceptance_layer": "protocol_smoke",
        "policy_kind": policy.kind,
        "policy_version": policy.version,
        "task_id": task.public.task_id,
        "model_outputs": len(outputs),
        "action_format_validity_rate": len(valid_outputs) / len(outputs) if outputs else None,
        "protocol_action_generated": bool(valid_outputs),
        "real_tool_call_executed": bool(tool_steps),
        "real_tool_call_observed": bool(
            tool_steps and all(step.tool_result is not None for step in tool_steps)
        ),
        "protocol_smoke_passed": bool(valid_outputs and tool_steps),
        "business_reward": verification,
        "verification_error": verification_error,
        "trajectory_path": str(path.relative_to(output)),
        "business_capability_not_inferred": True,
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["single", "protocol", "agent", "base", "base-benchmark"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen2.5-1.5b.toml")
    )
    parser.add_argument("--agent-config", type=Path, default=Path("configs/agent/qwen.toml"))
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/m2a/manifest.json"))
    parser.add_argument("--database-root", type=Path, default=Path("artifacts/m2a-databases"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    args = parser.parse_args()
    require_compute_node()
    # Refuse reused run directories; previous partial evidence must survive.
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = ModelSettings.load(args.model_config)
    settings = Settings.load(args.agent_config)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    metadata = {
        "status": "starting",
        "mode": args.mode,
        "split": args.split,
        "model_config": cfg.model_dump(),
        "settings": settings.model_dump(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "node": socket.gethostname(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
        "acceptance": {
            "infrastructure_smoke": "pending",
            "protocol_smoke": "not_run",
            "agent_capability_smoke": "not_run",
            "base_validation": "not_run",
        },
    }
    metadata_path = args.output / "execution.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for name, argv in [
        ("scheduler-job", ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]]),
        ("nvidia-smi", ["nvidia-smi"]),
    ]:
        probe = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        (args.output / (name + ".txt")).write_text(probe.stdout + probe.stderr)
    (args.output / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    log = (args.output / "server.log").open("w")
    process = None

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "enterprise_agent_rl.model_server",
                "--cache",
                str(args.cache),
                "--config",
                str(args.model_config),
                "--output",
                str(args.output / "server"),
                "--port",
                str(port),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 180
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"model_server_exited:{process.returncode}")
            try:
                with opener.open(endpoint + "/health", timeout=2) as response:
                    health = json.load(response)
                if health["metadata"]["pid"] != process.pid:
                    raise RuntimeError("health_endpoint_process_mismatch")
                metadata["model_load"] = health["metadata"]
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("model_server_startup_timeout") from None
                time.sleep(0.5)
        metadata["status"] = "running"
        metadata["acceptance"]["infrastructure_smoke"] = "model_loaded_http_ready"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        if args.mode == "single":
            request = urllib.request.Request(
                endpoint + "/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(
                    {
                        "model": cfg.model_id,
                        "messages": [
                            {"role": "user", "content": "用一句话说明你如何帮助分析企业数据。"}
                        ],
                        "max_tokens": 64,
                        "temperature": 0,
                        "seed": cfg.generation.seed,
                        "timeout_s": settings.inference_timeout_s,
                    }
                ).encode(),
            )
            with opener.open(request, timeout=settings.inference_timeout_s + 2) as response:
                result = json.load(response)
            write_json(args.output / "single-generation.json", result)
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            infrastructure_passed = bool(isinstance(content, str) and content.strip())
            metadata["acceptance"]["infrastructure_smoke"] = (
                "passed" if infrastructure_passed else "failed_empty_generation"
            )
            if not infrastructure_passed:
                raise RuntimeError("empty_output")
        elif args.mode == "protocol":
            policy = LocalModelPolicy(
                endpoint + "/v1/chat/completions",
                cfg.model_id,
                "qwen-base-" + cfg.revision[:12],
                cfg.generation,
            )
            protocol = run_protocol_smoke(policy, settings, args.output / "protocol")
            metadata["acceptance"]["protocol_smoke"] = (
                "passed" if protocol["protocol_smoke_passed"] else "failed_protocol_contract"
            )
        else:
            policy = LocalModelPolicy(
                endpoint + "/v1/chat/completions",
                cfg.model_id,
                "qwen-base-" + cfg.revision[:12],
                cfg.generation,
            )
            if args.mode == "agent":
                db = initialize_database(args.output / "smoke.sqlite3")
                tasks = [t for t in benchmark_tasks() if t.public.task_id == "top-2025"]
                report = evaluate(policy, tasks, db, settings, args.output / "smoke")
                if report["complete"] and report["metrics"]["task_success_rate"] == 1:
                    metadata["acceptance"]["agent_capability_smoke"] = "passed"
                elif report["complete"]:
                    metadata["acceptance"]["agent_capability_smoke"] = "business_failed"
                else:
                    metadata["acceptance"]["agent_capability_smoke"] = "incomplete_verification"
            elif args.mode == "base":
                report = evaluate_suite(
                    policy,
                    args.manifest,
                    args.database_root,
                    args.split,
                    settings,
                    args.output / "base",
                )
                metadata["acceptance"]["base_validation"] = (
                    "complete" if report["complete"] else "incomplete_verification"
                )
            else:
                report = run_base_benchmark(
                    policy,
                    args.manifest,
                    args.database_root,
                    settings,
                    cfg,
                    args.output.parent,
                    runtime_metadata=metadata.get("model_load"),
                )
                metadata["acceptance"]["base_validation"] = report["status"]
        metadata["status"] = "completed"
    except BaseException as exc:
        if args.mode == "base-benchmark":
            try:
                write_incomplete_report(
                    args.output.parent,
                    args.manifest,
                    args.database_root,
                    cfg,
                    settings,
                    {"type": type(exc).__name__, "message": str(exc)},
                )
            except Exception as report_exc:
                metadata["benchmark_finalize_error"] = {
                    "type": type(report_exc).__name__,
                    "message": str(report_exc),
                }
        metadata["acceptance"]["infrastructure_smoke"] = (
            "failed"
            if metadata["acceptance"]["infrastructure_smoke"]
            in {"pending", "model_loaded_http_ready"}
            else metadata["acceptance"]["infrastructure_smoke"]
        )
        metadata.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if process is not None:
            stop_model_process(process)
            metadata["server_exit_code"] = process.returncode
        log.close()
        metadata["server_cleaned_up"] = process is None or process.poll() is not None
        metadata["acceptance"]["resource_recovery"] = (
            "passed" if metadata["server_cleaned_up"] else "failed"
        )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        write_json(args.output / "acceptance.json", metadata)


if __name__ == "__main__":
    main()
