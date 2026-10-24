#!/usr/bin/env python3
"""Runtime performance sanity check; this file does not change production code.

The check deliberately keeps the frozen model/configuration and compares:

* raw Transformers FP16 + SDPA math;
* the current EnterpriseAgent QwenEngine with its finite-logit guard and event
  persistence;
* the same QwenEngine with only event persistence disabled for diagnosis.

It then uses that one loaded QwenEngine through the real loopback HTTP
endpoint for exactly three frozen validation tasks.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PROMPT = "Who are you? Answer in one short sentence."
MAX_NEW_TOKENS = 32
TASK_IDS = (
    "validation-0-schema",
    "validation-0-top",
    "validation-0-recovery",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(value: Any) -> Any:
    return copy.deepcopy(value)


class ForwardTrace:
    """Record cache evidence without changing model outputs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def install(self, model: Any) -> None:
        original = model.forward
        trace = self

        def wrapped(module: Any, *args: Any, **kwargs: Any) -> Any:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args and hasattr(args[0], "shape"):
                input_ids = args[0]
            past = kwargs.get("past_key_values")
            record = {
                "input_sequence_length": (
                    int(input_ids.shape[-1]) if input_ids is not None else None
                ),
                "past_key_values_present": past is not None,
                "past_key_values_type": type(past).__name__ if past is not None else None,
                "cache_position_present": kwargs.get("cache_position") is not None,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask_shape": (
                    list(kwargs["attention_mask"].shape)
                    if kwargs.get("attention_mask") is not None
                    else None
                ),
            }
            trace.calls.append(record)
            return original(*args, **kwargs)

        model.forward = types.MethodType(wrapped, model)

    def slice(self, start: int) -> dict[str, Any]:
        calls = self.calls[start:]
        cached = [item for item in calls if item["past_key_values_present"]]
        return {
            "forward_call_count": len(calls),
            "past_key_values_present_call_count": len(cached),
            "past_key_values_reused": bool(cached),
            "all_cached_calls_are_single_token": all(
                item["input_sequence_length"] == 1 for item in cached
            ),
            "calls": snapshot(calls),
        }


class GuardTrace:
    """Count the existing guard calls; the guard implementation remains intact."""

    def __init__(self, original_class: type) -> None:
        self.original_class = original_class
        self.records: list[dict[str, Any]] = []

    def replacement(self) -> type:
        outer = self
        original_class = self.original_class

        class CountingFiniteLogitsProcessor(original_class):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._diagnostic_record = {
                    "prompt_tokens": kwargs.get("prompt_tokens"),
                    "steps_checked": 0,
                }
                outer.records.append(self._diagnostic_record)

            def __call__(self, input_ids: Any, scores: Any) -> Any:
                self._diagnostic_record["steps_checked"] += 1
                return super().__call__(input_ids, scores)

        return CountingFiniteLogitsProcessor


def load_raw_model(cache: Path, model_config: Path) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = __import__("tomllib").loads(model_config.read_text())
    if config.get("model_id") != MODEL_ID or config.get("revision") != REVISION:
        raise RuntimeError("model_config_is_not_the_pinned_qwen_revision")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires_exactly_one_visible_cuda_device")
    kwargs = {
        "revision": REVISION,
        "cache_dir": str(cache),
        "local_files_only": True,
        "trust_remote_code": False,
    }
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            output_attentions=False,
            use_safetensors=True,
            **kwargs,
        )
        .to("cuda:0")
        .eval()
    )
    torch.cuda.synchronize()
    load_time = time.perf_counter() - started
    generation_config = model.generation_config.to_dict()
    metadata = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "model_class": type(model).__name__,
        "model_dtype": str(next(model.parameters()).dtype),
        "model_device": str(next(model.parameters()).device),
        "effective_attention": getattr(model.config, "_attn_implementation", None),
        "model_config_use_cache": getattr(model.config, "use_cache", None),
        "generation_config_use_cache": generation_config.get("use_cache"),
        "generation_config": generation_config,
        "output_attentions": getattr(model.config, "output_attentions", None),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "load_time_s": load_time,
        "load_count": 1,
        "tokenizer_load_count": 1,
        "load_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "load_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    return tokenizer, model, metadata


def prompt_inputs(tokenizer: Any, device: str) -> tuple[Any, str]:
    messages = [{"role": "user", "content": PROMPT}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    return inputs, rendered


def token_ids(tensor: Any) -> list[int]:
    return [int(item) for item in tensor.detach().cpu().reshape(-1).tolist()]


def raw_generate(tokenizer: Any, model: Any, output: Path) -> dict[str, Any]:
    import torch

    from enterprise_agent_rl.inference_stability import attention_backend_context

    inputs, rendered = prompt_inputs(tokenizer, "cuda:0")
    input_tokens = int(inputs.input_ids.shape[-1])
    trace = ForwardTrace()
    trace.install(model)
    generation_config_before = model.generation_config.to_dict()
    kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
        "output_attentions": False,
    }
    # The previous 44.40 tok/s diagnostic performed a finite SDPA forward
    # before its measured generate call. Preserve that measurement scope while
    # recording the warm-up explicitly; the forward itself is not counted in
    # generation latency or decode throughput.
    torch.cuda.synchronize()
    warmup_started = time.perf_counter()
    with attention_backend_context(torch, "sdpa_math"):
        with torch.inference_mode():
            warmup_output = model(**inputs, use_cache=True, output_attentions=False)
    torch.cuda.synchronize()
    warmup_latency = time.perf_counter() - warmup_started
    del warmup_output
    generation_trace_start = len(trace.calls)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with attention_backend_context(torch, "sdpa_math"):
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    all_ids = token_ids(generated[0])
    prompt_ids = token_ids(inputs.input_ids[0])
    generated_ids = all_ids[len(prompt_ids) :]
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_positions = [i for i, value in enumerate(generated_ids) if value in (eos_ids or [])]
    generation_config_after = model.generation_config.to_dict()
    result = {
        "label": "raw_transformers_sdpa_math",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "prompt": PROMPT,
        "chat_template_text": rendered,
        "input_tokens": input_tokens,
        "output_tokens": len(generated_ids),
        "requested_max_new_tokens": MAX_NEW_TOKENS,
        "generated_token_ids": generated_ids,
        "output": tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ),
        "eos_token_ids": eos_ids,
        "eos_positions": eos_positions,
        "finish_reason": "stop" if eos_positions else "length",
        "generation_kwargs": kwargs,
        "generation_latency_s": latency,
        "decode_tok_s": len(generated_ids) / latency if latency else None,
        "warmup_forward_latency_s": warmup_latency,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cache": {
            "model_config_use_cache": getattr(model.config, "use_cache", None),
            "generation_config_use_cache": generation_config_before.get("use_cache"),
            "generation_kwargs_use_cache": kwargs["use_cache"],
            **trace.slice(generation_trace_start),
        },
        "generation_config_unchanged": generation_config_before == generation_config_after,
        "generation_config_before": generation_config_before,
        "generation_config_after": generation_config_after,
    }
    write_json(output / "raw-sdpa.json", result)
    del generated, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def choose_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_model_server(
    cache: Path, model_config: Path, output: Path
) -> tuple[subprocess.Popen, str, Any]:
    """Start the unchanged production model server in its own process."""

    port = choose_port()
    output.mkdir(parents=True, exist_ok=True)
    log_stream = (output / "server.log").open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "enterprise_agent_rl.model_server",
            "--cache",
            str(cache),
            "--config",
            str(model_config),
            "--output",
            str(output),
            "--port",
            str(port),
        ],
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    endpoint = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with opener.open(endpoint + "/health", timeout=1) as response:
                if response.status == 200:
                    return process, endpoint, log_stream
        except OSError:
            if process.poll() is not None:
                break
            time.sleep(0.1)
    stop_model_server(process, log_stream)
    raise TimeoutError("runtime_sanity_model_server_startup_timeout")


def stop_model_server(process: subprocess.Popen, log_stream: Any) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    log_stream.close()


def http_json(
    endpoint: str, payload: dict[str, Any], timeout: float = 180
) -> tuple[dict, float, int]:
    request = urllib.request.Request(
        endpoint + "/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode(),
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response), time.perf_counter() - started, response.status
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read(1048576))
        return body, time.perf_counter() - started, exc.code


def runtime_direct_request(
    engine: Any,
    payload: dict[str, Any],
    label: str,
    output: Path,
    forward_trace: ForwardTrace,
    guard_trace: GuardTrace,
    *,
    disable_event_persistence: bool = False,
) -> dict[str, Any]:
    import enterprise_agent_rl.model_server as model_server
    from enterprise_agent_rl.model_server import CompletionRequest

    original_append_event = model_server.append_event
    if disable_event_persistence:
        model_server.append_event = lambda path, event: None
    forward_start = len(forward_trace.calls)
    guard_start = len(guard_trace.records)
    started = time.perf_counter()
    try:
        response = engine.generate(CompletionRequest.model_validate(payload))
    finally:
        model_server.append_event = original_append_event
    wall_latency = time.perf_counter() - started
    telemetry = response.get("telemetry", {})
    result = {
        "label": label,
        "request": payload,
        "http_status": None,
        "direct_wall_latency_s": wall_latency,
        "response": response,
        "input_tokens": telemetry.get("input_tokens"),
        "output_tokens": telemetry.get("output_tokens"),
        "generation_latency_s": telemetry.get("generation_latency_s"),
        "decode_tok_s": telemetry.get("generation_tokens_per_s"),
        "peak_allocated_bytes": telemetry.get("peak_allocated_bytes"),
        "peak_reserved_bytes": telemetry.get("peak_reserved_bytes"),
        "event_persistence_disabled": disable_event_persistence,
        "cache": forward_trace.slice(forward_start),
        "guard": snapshot(guard_trace.records[guard_start:]),
    }
    write_json(output / f"{label}.json", result)
    return result


def recovery_count(trajectory: Any) -> int:
    sql_steps = [step for step in trajectory.steps if step.tool_name == "sql"]
    return sum(
        any(next_step.tool_result.ok for next_step in sql_steps[index + 1 :])
        for index, step in enumerate(sql_steps)
        if not step.tool_result.ok
    )


def smoke_task_records(report: dict[str, Any], smoke_output: Path, database_id: str) -> list[dict]:
    from enterprise_agent_rl.trajectories import TrajectoryStore

    records = []
    store = TrajectoryStore(smoke_output / "trajectories")
    for episode in report["episodes"]:
        trajectory = store.load(smoke_output / episode["trajectory_path"])
        attempts = trajectory.model_attempts
        tools = [step for step in trajectory.steps if step.tool_name]
        sql_steps = [step for step in trajectory.steps if step.tool_name == "sql"]
        input_tokens = sum(a.get("usage", {}).get("prompt_tokens", 0) for a in attempts)
        output_tokens = sum(a.get("usage", {}).get("completion_tokens", 0) for a in attempts)
        reward = trajectory.reward.model_dump() if trajectory.reward else None
        records.append(
            {
                "task_id": trajectory.task.task_id,
                "database": database_id,
                "success": bool(reward and reward.get("success")),
                "failure_reason": (
                    None
                    if reward and reward.get("success")
                    else (reward or {}).get("reason") or trajectory.terminal_reason
                ),
                "tool_calls": len(tools),
                "invalid_tool_calls": sum(not step.tool_result.ok for step in tools),
                "SQL_attempts": len(sql_steps),
                "recovery_count": recovery_count(trajectory),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency": trajectory.elapsed_s,
                "reward": reward,
                "trajectory_path": str((smoke_output / episode["trajectory_path"]).resolve()),
                "model_attempt_count": len(attempts),
                "policy_kind": trajectory.policy_kind,
                "policy_version": trajectory.policy_version,
                "terminal_reason": trajectory.terminal_reason,
            }
        )
    return records


def run_agent_smoke(
    endpoint: str,
    cfg: Any,
    settings: Any,
    manifest: Path,
    database_root: Path,
    output: Path,
) -> dict[str, Any]:
    from enterprise_agent_rl.agent import LocalModelPolicy
    from enterprise_agent_rl.benchmark import load_frozen_manifest
    from enterprise_agent_rl.data import BenchmarkTask
    from enterprise_agent_rl.evaluation import evaluate

    frozen = load_frozen_manifest(manifest, database_root)
    selected = [
        record for record in frozen["records"] if record["task"]["public"]["task_id"] in TASK_IDS
    ]
    if [record["task"]["public"]["task_id"] for record in selected] != list(TASK_IDS):
        raise RuntimeError("validation_task_selection_mismatch")
    if any(record["task"]["split"] != "validation" for record in selected):
        raise RuntimeError("non_validation_task_selected")
    databases = {record["database_id"] for record in selected}
    if databases != {"validation-0"}:
        raise RuntimeError("smoke_database_selection_mismatch")
    database = database_root / "validation-0.sqlite3"
    database_before = sha256_file(database)
    tasks = [BenchmarkTask.model_validate(record["task"]) for record in selected]
    policy = LocalModelPolicy(
        endpoint + "/v1/chat/completions",
        cfg.model_id,
        "qwen-base-runtime-sanity-" + cfg.revision[:12],
        cfg.generation,
    )
    report = evaluate(policy, tasks, database, settings, output)
    database_after = sha256_file(database)
    if database_before != database_after:
        raise RuntimeError("validation_database_changed")
    details = smoke_task_records(report, output, "validation-0")
    result = {
        "split": "validation",
        "test_split_used": False,
        "train_split_used": False,
        "task_ids": list(TASK_IDS),
        "database": "validation-0",
        "database_sha256_before": database_before,
        "database_sha256_after": database_after,
        "manifest_sha256": frozen["manifest_sha256"],
        "policy_kind": report["metrics"]["policy_kind"] if report.get("metrics") else None,
        "policy_version": report["metrics"]["policy_version"] if report.get("metrics") else None,
        "real_tool_execution": True,
        "verifier_used": True,
        "reward_used": True,
        "deterministic_fixture_expert_used": False,
        "report": report,
        "tasks": details,
    }
    write_json(output.parent / "agent-smoke-summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("artifacts/model-cache"))
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen2.5-1.5b.toml")
    )
    parser.add_argument("--agent-config", type=Path, default=Path("configs/agent/qwen.toml"))
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/m2a/manifest.json"))
    parser.add_argument("--database-root", type=Path, default=Path("artifacts/m2a-databases"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "started",
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], text=True)
        .strip(),
        "git_status": __import__("subprocess").check_output(
            ["git", "status", "--short", "--branch"], text=True
        ),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "no_checkpoint_or_generation_config_write": True,
        "test_split_used": False,
        "train_split_used": False,
    }
    write_json(args.output / "progress.json", report)
    server_process = None
    server_log = None
    engine = None
    try:
        import torch

        import enterprise_agent_rl.model_server as model_server
        from enterprise_agent_rl.common import Settings
        from enterprise_agent_rl.model_server import ModelSettings, QwenEngine

        cfg = ModelSettings.load(args.model_config)
        settings = Settings.load(args.agent_config)
        report["model_config"] = cfg.model_dump()
        report["agent_settings"] = settings.model_dump()
        report["runtime"] = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        }

        # Raw model is independent of the EnterpriseAgent server.
        raw_tokenizer, raw_model, raw_load = load_raw_model(args.cache, args.model_config)
        raw_trace = raw_generate(raw_tokenizer, raw_model, args.output)
        report["raw"] = {"load": raw_load, "generation": raw_trace}
        del raw_model, raw_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # Load the current runtime engine once for the direct performance A/B.
        # The unchanged production server is loaded once separately for the
        # real Agent smoke below, after this diagnostic engine is freed.
        runtime_output = args.output / "runtime"
        runtime_output.mkdir(parents=True, exist_ok=True)
        guard_trace = GuardTrace(model_server.FiniteLogitsProcessor)
        model_server.FiniteLogitsProcessor = guard_trace.replacement()
        engine = QwenEngine(cfg, args.cache, runtime_output)
        forward_trace = ForwardTrace()
        forward_trace.install(engine.model)
        runtime_load = snapshot(engine.metadata)
        report["runtime_load"] = runtime_load
        report["runtime_load_counts"] = {"model": 1, "tokenizer": 1}
        report["runtime_cache_config"] = {
            "model_config_use_cache": getattr(engine.model.config, "use_cache", None),
            "generation_config_use_cache": engine.model.generation_config.to_dict().get(
                "use_cache"
            ),
            "production_generate_use_cache_argument": True,
        }

        request = {
            "model": cfg.model_id,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": MAX_NEW_TOKENS,
            "seed": 17,
            "timeout_s": 120.0,
            "stream": False,
        }
        a = runtime_direct_request(
            engine,
            request,
            "enterprise-runtime-sdpa-guard-telemetry",
            args.output,
            forward_trace,
            guard_trace,
        )
        c = runtime_direct_request(
            engine,
            request,
            "enterprise-runtime-sdpa-guard-no-event-persistence",
            args.output,
            forward_trace,
            guard_trace,
            disable_event_persistence=True,
        )

        report["performance"] = {
            "raw": raw_trace,
            "runtime": a,
            "runtime_guard_no_event_persistence": c,
            "telemetry_overhead": {
                "generation_latency_delta_s": (
                    a.get("generation_latency_s") - c.get("generation_latency_s")
                    if a.get("generation_latency_s") is not None
                    and c.get("generation_latency_s") is not None
                    else None
                ),
                "direct_wall_delta_s": a["direct_wall_latency_s"] - c["direct_wall_latency_s"],
                "interpretation": (
                    "A vs C isolates event JSON/fsync persistence; guard remains enabled in both."
                ),
            },
            "runtime_generation_calls_before_agent_smoke": 2,
            "runtime_model_load_count": 1,
            "runtime_tokenizer_load_count": 1,
        }
        write_json(args.output / "performance-summary.json", report["performance"])
        write_json(
            args.output / "cache-and-guard-summary.json",
            {
                "raw": raw_trace["cache"],
                "runtime_guard_telemetry": {"cache": a["cache"], "guard": a["guard"]},
                "runtime_guard_no_event_persistence": {"cache": c["cache"], "guard": c["guard"]},
            },
        )
        write_json(args.output / "progress.json", report)

        # Free the diagnostic engine before starting the unchanged production
        # server for the real Agent smoke. That server stays alive across all
        # three tasks, so its model/tokenizer load count is one.
        del engine
        engine = None
        gc.collect()
        torch.cuda.empty_cache()
        agent_server_output = args.output / "agent-server"
        server_process, endpoint, server_log = start_model_server(
            args.cache, args.model_config, agent_server_output
        )
        agent_output = args.output / "agent-smoke"
        smoke = run_agent_smoke(
            endpoint,
            cfg,
            settings,
            args.manifest,
            args.database_root,
            agent_output,
        )
        report["agent_smoke"] = smoke
        report["runtime_generation_calls_total"] = 2 + sum(
            item["model_attempt_count"] for item in smoke["tasks"]
        )
        generation_log = agent_server_output / "generations.jsonl"
        report["agent_server_load_counts"] = {
            "model": 1,
            "tokenizer": 1,
            "generation_calls": sum(1 for _ in generation_log.open())
            if generation_log.is_file()
            else 0,
        }
        report["agent_server_model_load"] = json.loads(
            (agent_server_output / "model-load.json").read_text()
        )
        report["runtime_guard_records"] = snapshot(guard_trace.records)
        report["runtime_forward_call_count_total"] = len(forward_trace.calls)
        report["status"] = "complete"
        report["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        write_json(args.output / "sanity-report.json", report)
        return 0
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        write_json(args.output / "sanity-report.json", report)
        raise
    finally:
        if server_process is not None and server_log is not None:
            stop_model_server(server_process, server_log)
        if engine is not None:
            del engine


if __name__ == "__main__":
    raise SystemExit(main())
