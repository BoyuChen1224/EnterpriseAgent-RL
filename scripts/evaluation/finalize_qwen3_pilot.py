#!/usr/bin/env python3
"""Assemble the immutable T-031/E-028 run handoff after both GPU stages."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def _read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name.endswith(".tmp") or path.name == "manifest.json":
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    return records


def _case_table(metrics: dict[str, Any], case_path: Path) -> str:
    rows = [
        "| task | category | scored | unknown | success | model requests | tool attempts |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if case_path.is_file():
        for line in case_path.read_text().splitlines():
            case = json.loads(line)
            rows.append(
                "| {task_id} | {category} | {scored} | {unknown} | {success} | "
                "{model_attempts} | {tool_attempts} |".format(**case)
            )
    else:
        rows.append("| no case results | | | | | | |")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tests-exit-code", type=int, default=0)
    parser.add_argument("--cache-download-exit-code", type=int, default=0)
    parser.add_argument("--cache-verification-exit-code", type=int, default=0)
    args = parser.parse_args()
    run = args.run
    base_metrics = _read(run / "base" / "metrics.json", {})
    lora_metrics = _read(run / "lora" / "training-metrics.json", {})
    reload_check = _read(run / "lora" / "reload-check.json", {})
    allocation = _read(run / "allocation-ledger.json", {})
    job_chain = _read(run / "job-chain.json", {})
    cache_check = _read(run / "cache-verification.json", {})
    panel = _read(run / "panel-manifest.json", {})
    lora_checks = _read(run / "checks.json", {})

    base_complete = base_metrics.get("status") == "BASE_PANEL_COMPLETE"
    lora_pass = reload_check.get("status") == "LORA_UPDATE_RELOAD_PASS" and (
        lora_metrics.get("successful_optimizer_updates") == 2
    )
    checks = {
        **(lora_checks if isinstance(lora_checks, dict) else {}),
        "base_panel_complete": base_complete,
        "base_unknown_zero": base_metrics.get("counts", {}).get("unknown") == 0,
        "base_scored_planned": base_metrics.get("counts", {}).get("scored")
        == panel.get("planned", 8),
        "lora_update_reload_pass": lora_pass,
        "lora_successful_optimizer_updates": lora_metrics.get("successful_optimizer_updates"),
        "lora_module_count_252": lora_metrics.get("module_count") == 252,
        "lora_backward_within_24": lora_metrics.get("backward_count", 999) <= 24,
        "tests_exit_code": args.tests_exit_code,
        "cache_verification_exit_code": args.cache_verification_exit_code,
        "allocation_within_240_gpu_minutes": allocation.get("within_budget"),
        "status": "T031_E028_PASS"
        if base_complete and lora_pass and args.tests_exit_code == 0
        else "T031_E028_INCOMPLETE",
    }
    _write(run / "checks.json", checks)

    commands = [
        {
            "name": "train_only_database_prepare",
            "command": (
                ".venv/bin/python scripts/evaluation/run_qwen3_base_panel.py "
                "--output artifacts/qwen3-base-lora-pilot/t031-v1 --prepare"
            ),
            "exit_code": 0,
            "scope": "cpu",
        },
        {
            "name": "isolated_environment_and_cpu_regression",
            "command": (
                ".venv-qwen3/bin/pytest -q tests/test_qwen3_pilot.py tests/test_qwen_protocol.py "
                "tests/test_sft_training.py tests/test_inference_stability.py"
            ),
            "exit_code": args.tests_exit_code,
            "scope": "cpu",
        },
        {
            "name": "qwen3_pinned_snapshot_download_initial",
            "command": (
                ".venv-qwen3/bin/python scripts/setup/cache_qwen.py "
                "--cache artifacts/model-cache-qwen3-8b --config configs/model/qwen3-8b.toml"
            ),
            "exit_code": 1,
            "failure_class": "network_chunked_encoding_incomplete_read",
            "log": "logs/cache-download.log",
            "scope": "login",
        },
        {
            "name": "qwen3_pinned_snapshot_download_retry_1",
            "command": (
                ".venv-qwen3/bin/python scripts/setup/cache_qwen.py "
                "--cache artifacts/model-cache-qwen3-8b --config configs/model/qwen3-8b.toml"
            ),
            "exit_code": 1,
            "failure_class": "network_tls_unexpected_eof",
            "log": "logs/cache-download-retry-1.log",
            "scope": "login",
        },
        {
            "name": "qwen3_pinned_snapshot_download_retry_2",
            "command": (
                ".venv-qwen3/bin/python scripts/setup/cache_qwen.py "
                "--cache artifacts/model-cache-qwen3-8b --config configs/model/qwen3-8b.toml"
            ),
            "exit_code": 1,
            "failure_class": "network_tls_unexpected_eof",
            "log": "logs/cache-download-retry-2.log",
            "scope": "login",
        },
        {
            "name": "qwen3_pinned_snapshot_download_retry_3",
            "command": (
                ".venv-qwen3/bin/python scripts/setup/cache_qwen.py "
                "--cache artifacts/model-cache-qwen3-8b --config configs/model/qwen3-8b.toml"
            ),
            "exit_code": args.cache_download_exit_code,
            "failure_class": None,
            "log": "logs/cache-download-retry-3.log",
            "scope": "login",
        },
        {
            "name": "qwen3_cache_verification",
            "command": (
                ".venv-qwen3/bin/python scripts/setup/verify_qwen_cache.py "
                "--cache artifacts/model-cache-qwen3-8b --config configs/model/qwen3-8b.toml "
                "--output artifacts/qwen3-base-lora-pilot/t031-v1/cache-verification.json"
            ),
            "exit_code": args.cache_verification_exit_code,
            "log": "logs/cache-verification-final.log",
            "scope": "login",
        },
        {
            "name": "qwen3_cached_tokenizer_hard_off_smoke",
            "command": "inline cached tokenizer smoke; see tokenizer-preflight.json",
            "exit_code": 0,
            "log": "tokenizer-preflight.json",
            "scope": "login",
        },
        *[
            {
                "name": f"slurm_{item.get('stage')}",
                "job_id": item.get("job_id"),
                "state": item.get("state"),
                "exit_code": item.get("exit_code"),
                "command": item.get("command"),
                "scope": "gpu",
            }
            for item in job_chain.get("records", [])
        ],
    ]
    _write(run / "commands.json", {"schema_version": "1.0", "commands": commands})

    environment = _read(run / "environment.json", {})
    if not isinstance(environment, dict):
        environment = {}
    environment.update(
        {
            "experiment": "T-031/E-028",
            "model_id": "Qwen/Qwen3-8B",
            "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "base_workers": [
                _read(path / "worker-meta.json", {})
                for path in sorted((run / "base").glob("worker-*/"))
                if (path / "worker-meta.json").is_file()
            ],
            "cache_verification": cache_check,
            "git_commit_at_finalize": _git("rev-parse", "HEAD"),
            "git_dirty_at_finalize": bool(_git("status", "--porcelain")),
        }
    )
    _write(run / "environment.json", environment)

    frozen = _read(run / "frozen-config.json", {})
    if not isinstance(frozen, dict):
        frozen = {}
    frozen.update(
        {
            "experiment": "T-031/E-028",
            "base_adapter": "NONE",
            "base_model": {
                "model_id": "Qwen/Qwen3-8B",
                "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                "dtype": "float16",
                "attention_backend": "sdpa_math",
                "enable_thinking": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 512,
                "seed": 17,
                "format_retries": 1,
                "parse_mode": "json_fence",
                "max_context_tokens": 4096,
            },
            "base_panel_results_sealed_before_lora": base_complete,
        }
    )
    _write(run / "frozen-config.json", frozen)

    preflight = {
        "schema_version": "1.0",
        "status": "passed" if base_complete and lora_pass else "incomplete",
        "cpu": _read(run / "cpu-preflight.json", {}),
        "cache": cache_check,
        "base": base_metrics,
        "lora": _read(run / "lora" / "lora-preflight.json", {}),
        "reload": reload_check,
    }
    _write(run / "preflight.json", preflight)
    tool_summary = (
        f"{base_metrics.get('tool_calls')}/{base_metrics.get('tool_errors')}"
    )

    report = f"""# T-031 / E-028 — Qwen3-8B Base and LoRA pilot

Status: **{checks['status']}**

This run uses `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`,
raw safetensors, FP16, SDPA math, and hard `enable_thinking=false`. The Base stage
is an 8-item train-only diagnostic panel; it is not compared with the old 20-item
Qwen2.5 validation result.

## Base panel

{_case_table(base_metrics, run / 'base' / 'case-results.jsonl')}

Counts: `{json.dumps(base_metrics.get('counts', {{}}), ensure_ascii=False)}`  
Business success rate (planned denominator): `{base_metrics.get('business_success_rate')}`  
SQL executable rate: `{base_metrics.get('sql_executable_rate')}`  
Engine events: `{base_metrics.get('engine_events')}`; tool calls/errors: `{tool_summary}`  
Status: `{base_metrics.get('status')}`. Business failures remain scored; only explicit
infrastructure/evidence failures are unknown.

## LoRA smoke

Selected `{lora_metrics.get('sample_count')}` re-tokenized train decision examples;
processed `{lora_metrics.get('microbatches_processed')}` microbatches with
`{lora_metrics.get('successful_optimizer_updates')}` successful optimizer updates.
Target modules: `{lora_metrics.get('module_count')}` (expected 252); base changed:
`{lora_metrics.get('base_parameters_changed')}`; LoRA changed tensors:
`{lora_metrics.get('lora_changed_parameter_count')}`; reload: `{reload_check.get('status')}`.

## Resource and evidence handoff

GPU allocation: `{allocation.get('total_gpu_minutes')}` minutes, within 240-minute cap:
`{allocation.get('within_budget')}`. Job chain is in `job-chain.json`; exact command
statuses are in `commands.json`. All request/response/engine events, SQL audits,
trajectories and verifier summaries remain under `base/`. Adapter, optimizer/scaler/RNG
checkpoint and reload evidence remain under `lora/`.

Generated at `{_now()}`. No post-training business panel, GRPO, validation or test
evaluation was run.
"""
    (run / "report.md").write_text(report)

    _write(
        run / "manifest.json",
        {
            "schema_version": "1.0",
            "experiment": "T-031/E-028",
            "created_at_utc": _now(),
            "status": checks["status"],
            "files": _file_manifest(run),
        },
    )
    return 0 if checks["status"] == "T031_E028_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
