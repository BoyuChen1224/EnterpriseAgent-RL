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

from enterprise_agent_rl.agent import LocalModelPolicy
from enterprise_agent_rl.base_evaluation import evaluate_suite
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import benchmark_tasks, initialize_database
from enterprise_agent_rl.evaluation import evaluate
from enterprise_agent_rl.gpu_guard import require_compute_node
from enterprise_agent_rl.model_server import ModelSettings


def stop_model_process(process):
    """Terminate only the process group created by this supervisor."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["single", "agent", "base"])
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
            (args.output / "single-generation.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            )
            if not result["choices"][0]["message"]["content"].strip():
                raise RuntimeError("empty_output")
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
                if report["metrics"]["task_success_rate"] != 1:
                    raise RuntimeError("agent_smoke_task_failed:see_real_trajectory")
            else:
                report = evaluate_suite(
                    policy,
                    args.manifest,
                    args.database_root,
                    args.split,
                    settings,
                    args.output / "base",
                )
                if not report["complete"]:
                    raise RuntimeError("incomplete_base_evaluation:see_verification_errors")
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if process is not None:
            stop_model_process(process)
            metadata["server_exit_code"] = process.returncode
        log.close()
        metadata["server_cleaned_up"] = process is None or process.poll() is not None
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
