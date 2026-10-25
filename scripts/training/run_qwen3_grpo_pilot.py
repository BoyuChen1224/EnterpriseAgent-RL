#!/usr/bin/env python3
"""T-034/E-031 fixed-batch Qwen3 GRPO update and comparison driver.

The driver has no rollout path.  It builds one immutable training batch from
the audited T-033 records, performs exactly two logical updates on a fresh
Base→parent-SFT model, freezes the resulting delta, and then runs a separate
SFT/RL validation comparison.  Every GPU stage is Slurm-only and writes its
own evidence before returning.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_DEFAULT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DEFAULT / "src"))

from enterprise_agent_rl.grpo.budget import BudgetError, BudgetGuard  # noqa: E402
from enterprise_agent_rl.grpo.config import RunContext  # noqa: E402
from enterprise_agent_rl.grpo.io import atomic_write_json, sha256_file  # noqa: E402
from enterprise_agent_rl.grpo.qwen3_rollout import (  # noqa: E402
    QWEN3_MODEL_ID,
    QWEN3_PARENT_CONFIG_SHA256,
    QWEN3_PARENT_WEIGHT_SHA256,
    QWEN3_REVISION,
)
from enterprise_agent_rl.grpo.qwen3_update import (  # noqa: E402
    EXPECTED_ACTIVE_RECORDS,
    EXPECTED_ACTIVE_TOKENS,
    EXPECTED_MAX_SEQUENCE,
    EXPECTED_MODULE_COUNT,
    EXPECTED_TOKEN_ROWS,
    EXPECTED_TOTAL_TOKENS,
    EXPECTED_ZERO_RECORDS,
    SOURCE_ARTIFACT_MANIFEST_SHA256,
    SOURCE_CASE_RESULTS_SHA256,
    SOURCE_GROUP_METRICS_SHA256,
    SOURCE_TOKEN_ROLLOUTS_SHA256,
    build_frozen_train_batch,
    load_qwen3_update_config,
    validate_train_batch,
)

RUN_DEFAULT = Path("artifacts/qwen3-grpo-update-pilot/t034-v1")
CONFIG_DEFAULT = Path("configs/training/qwen3-grpo-update-pilot.toml")
MODEL_CONFIG_DEFAULT = Path("configs/model/qwen3-8b.toml")
AGENT_CONFIG_DEFAULT = Path("configs/agent/qwen3-base-panel.toml")
T032_VALIDATION = Path("artifacts/qwen3-runtime-sft-comparison/t032-v1/evaluation/validation")
T032_VALIDATION_MANIFEST_SHA256 = "056777038235f49cd1f553879ddfd5ebe8379f77e18a8da46804f9dc28e47c2a"

SOURCE_PATHS = (
    "configs/training/qwen3-grpo-update-pilot.toml",
    "scripts/evaluation/run_qwen3_grpo_rollout.py",
    "scripts/evaluation/run_qwen3_runtime_panel.py",
    "scripts/slurm/qwen3_grpo_update_pilot.sbatch",
    "scripts/training/run_qwen3_grpo_pilot.py",
    "src/enterprise_agent_rl/grpo/qwen3_rollout.py",
    "src/enterprise_agent_rl/grpo/qwen3_update.py",
    "src/enterprise_agent_rl/grpo/torch_backend.py",
    "src/enterprise_agent_rl/grpo/upstream.py",
    "src/enterprise_agent_rl/inference_stability.py",
    "src/enterprise_agent_rl/lora.py",
    "src/enterprise_agent_rl/model_protocol.py",
    "src/enterprise_agent_rl/model_server.py",
    "tests/test_qwen3_grpo_update_pilot.py",
)


class PilotError(RuntimeError):
    """A fixed T-034 evidence or identity contract was violated."""


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PilotError(f"jsonl_missing:{path}")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _append_jsonl(path: Path, value: dict[str, Any], *, unique_key: str | None = None) -> None:
    existing = _read_jsonl(path) if path.is_file() else []
    if unique_key is not None and any(
        row.get(unique_key) == value.get(unique_key) for row in existing
    ):
        raise PilotError(f"duplicate_jsonl_record:{path}:{unique_key}:{value.get(unique_key)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _ensure_run_dir(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    owner_path = run_dir / "ownership.json"
    if owner_path.is_file():
        owner = _read_json(owner_path)
        if not isinstance(owner, dict) or owner.get("task") != "T-034/E-031":
            raise PilotError(f"run_dir_owned_by_other_task:{run_dir}")
        return
    entries = [item for item in run_dir.iterdir() if item.name not in {".gitkeep"}]
    if entries:
        raise PilotError(f"run_dir_exists_without_t034_ownership:{run_dir}")
    _write_json(
        owner_path,
        {
            "schema_version": "t034-run-ownership-v1",
            "task": "T-034/E-031",
            "experiment": "fixed Qwen3 two-update GRPO pilot",
            "created_at_utc": _now(),
            "owner_session": os.environ.get("CODEX_SESSION_ID"),
        },
    )


def _context(repo: Path, run_dir: Path, config_path: Path) -> RunContext:
    return RunContext(repo.resolve(), run_dir.resolve(), config_path.resolve())


def _config(repo: Path, run_dir: Path, config_path: Path, *, final_delta: Path | None = None):
    return load_qwen3_update_config(
        config_path,
        _context(repo, run_dir, config_path),
        final_delta=final_delta,
    )


def _resolved(path_value: str, repo: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo / path


def _assert_parent(repo: Path, config: Any) -> dict[str, Any]:
    parent = _resolved(str(config.model["parent_adapter"]), repo)
    weight = parent / "adapter_model.safetensors"
    adapter_config = parent / "adapter_config.json"
    if not weight.is_file() or sha256_file(weight) != QWEN3_PARENT_WEIGHT_SHA256:
        raise PilotError("parent_adapter_weight_identity_mismatch")
    if not adapter_config.is_file() or sha256_file(adapter_config) != QWEN3_PARENT_CONFIG_SHA256:
        raise PilotError("parent_adapter_config_identity_mismatch")
    return {
        "path": str(parent.resolve()),
        "adapter_model_sha256": sha256_file(weight),
        "adapter_config_sha256": sha256_file(adapter_config),
    }


def _environment(repo: Path, run_dir: Path, config: Any, *, stage: str) -> dict[str, Any]:
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "pydantic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema_version": "t034-environment-v1",
        "stage": stage,
        "created_at_utc": _now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": str(Path.cwd().resolve()),
        "repo_root": str(repo.resolve()),
        "run_dir": str(run_dir.resolve()),
        "model_id": config.model["model_id"],
        "model_revision": config.model["revision"],
        "dtype": config.model["dtype"],
        "attention_backend": config.model["attention_backend"],
        "thinking": config.model["enable_thinking"],
        "runtime_versions": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "validation_payload_read_by_training": False,
        "test_payload_accessed": False,
    }


def _source_snapshot(repo: Path, run_dir: Path) -> dict[str, Any]:
    root = run_dir / "source-snapshot"
    root.mkdir(parents=True, exist_ok=True)
    files = []
    patch_parts = []
    for relative in SOURCE_PATHS:
        source = repo / relative
        if not source.is_file():
            raise PilotError(f"source_snapshot_missing:{relative}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "snapshot_path": str(target.relative_to(run_dir)),
            }
        )
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--", relative],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if diff.stdout:
            patch_parts.append(diff.stdout)
        listed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if listed.returncode != 0:
            untracked = subprocess.run(
                ["git", "diff", "--no-index", "--", "/dev/null", relative],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            if untracked.stdout:
                patch_parts.append(untracked.stdout)
    patch = root / "dirty.patch"
    patch.write_text("\n".join(patch_parts), encoding="utf-8")
    status = subprocess.run(
        ["git", "status", "--short", "--", *SOURCE_PATHS],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    manifest = {
        "schema_version": "t034-source-snapshot-v1",
        "task": "T-034/E-031",
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "task_path_status": status,
        "source_paths": files,
        "dirty_patch": str(patch.relative_to(run_dir)),
        "dirty_patch_sha256": sha256_file(patch),
        "protected_paths_included": [],
        "created_at_utc": _now(),
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def _initial_ledger(run_dir: Path) -> None:
    ledger = run_dir / "allocation-ledger.json"
    if not ledger.is_file():
        _write_json(
            ledger,
            {
                "schema_version": "t034-allocation-ledger-v1",
                "task": "T-034/E-031",
                "records": [],
                "total_gpu_seconds": 0.0,
                "total_gpu_minutes": 0.0,
                "budget_gpu_seconds": 10800,
                "old_t033_gpu_minutes_excluded": 15.633333333333334,
            },
        )
    chain = run_dir / "job-chain.json"
    if not chain.is_file():
        _write_json(
            chain,
            {
                "schema_version": "t034-job-chain-v1",
                "task": "T-034/E-031",
                "jobs": [],
                "at_most_one_executable_job": True,
                "pending_replacements": [],
            },
        )


def _prepare(repo: Path, run_dir: Path, config_path: Path) -> dict[str, Any]:
    _ensure_run_dir(run_dir)
    config, context = _config(repo, run_dir, config_path)
    if config.model["model_id"] != QWEN3_MODEL_ID or config.model["revision"] != QWEN3_REVISION:
        raise PilotError("t034_model_identity_mismatch")
    parent = _assert_parent(repo, config)
    source = _resolved(str(config.data["source_run"]), repo)
    batch_path = run_dir / "training" / "inputs" / "train-batch.jsonl"
    batch_summary = build_frozen_train_batch(source, batch_path)
    expected_hash = batch_summary["output_sha256"]
    if sha256_file(batch_path) != expected_hash:
        raise PilotError("frozen_train_batch_hash_mismatch_after_write")
    counts = batch_summary["counts"]
    if counts != {
        "records": EXPECTED_TOKEN_ROWS,
        "active_records": EXPECTED_ACTIVE_RECORDS,
        "zero_records": EXPECTED_ZERO_RECORDS,
        "total_tokens": EXPECTED_TOTAL_TOKENS,
        "active_tokens": EXPECTED_ACTIVE_TOKENS,
        "max_full_sequence_tokens": EXPECTED_MAX_SEQUENCE,
    }:
        raise PilotError(f"frozen_train_batch_counts_mismatch:{counts}")
    _write_json(run_dir / "frozen-config.json", {"config": config.raw, "task": "T-034/E-031"})
    _write_json(run_dir / "environment.json", _environment(repo, run_dir, config, stage="prepare"))
    _write_json(
        run_dir / "training" / "input-manifest.json",
        {
            "schema_version": "t034-input-manifest-v1",
            "task": "T-034/E-031",
            "source_run": str(source),
            "source_hashes": {
                "artifact_manifest": SOURCE_ARTIFACT_MANIFEST_SHA256,
                "token_rollouts": SOURCE_TOKEN_ROLLOUTS_SHA256,
                "group_metrics": SOURCE_GROUP_METRICS_SHA256,
                "case_results": SOURCE_CASE_RESULTS_SHA256,
            },
            "source_manifest_bindings": {
                "artifact_manifest": str(source / "artifact-manifest.json"),
                "token_rollouts": str(source / "token-rollouts.jsonl"),
                "group_metrics": str(source / "group-metrics.json"),
                "case_results": str(source / "case-results.jsonl"),
            },
            "batch_path": str(batch_path.resolve()),
            "batch_sha256": expected_hash,
            "counts": counts,
            "input_cap_tokens": 4096,
            "original_advantage_all_null": True,
            "old_logps_fixed": True,
            "zero_advantage_records_retained": True,
            "validation_payload_read_by_training": False,
            "parent_dependency": parent,
        },
    )
    _write_json(
        run_dir / "training" / "preflight.json",
        {
            "schema_version": "t034-preflight-v1",
            "status": "PASSED",
            "source_chain": batch_summary["source_chain"],
            "counts": counts,
            "advantage_definition": batch_summary["advantage_definition"],
            "group_summary": batch_summary["groups"],
            "model_load_count": 0,
            "rollout_generation_count": 0,
            "validation_payload_read": False,
            "test_payload_accessed": False,
        },
    )
    _initial_ledger(run_dir)
    _source_snapshot(repo, run_dir)
    pilot_command = f"{sys.executable} scripts/training/run_qwen3_grpo_pilot.py"
    _write_json(
        run_dir / "commands.json",
        {
            "schema_version": "t034-commands-v1",
            "commands": {
                "prepare": f"{pilot_command} --stage prepare --output {run_dir}",
                "tiny": f"{pilot_command} --stage tiny --output {run_dir}",
                "train": "T034_STAGE=train scripts/slurm/qwen3_grpo_update_pilot.sbatch",
                "reload": "T034_STAGE=reload scripts/slurm/qwen3_grpo_update_pilot.sbatch",
                "compare": "T034_STAGE=compare scripts/slurm/qwen3_grpo_update_pilot.sbatch",
                "finalize": f"{pilot_command} --stage finalize --output {run_dir}",
            },
            "created_at_utc": _now(),
        },
    )
    return {"status": "PREPARED", "counts": counts, "batch_sha256": expected_hash}


def _run_logged(
    command: list[str], *, cwd: Path, env: dict[str, str], stdout_path: Path, stderr_path: Path
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=False
        )
    return int(completed.returncode)


def _tiny(repo: Path, run_dir: Path, config_path: Path) -> dict[str, Any]:
    _ensure_run_dir(run_dir)
    batch = _read_jsonl(run_dir / "training" / "inputs" / "train-batch.jsonl")
    observed = validate_train_batch(batch)
    bad = dict(batch[0])
    bad["advantage"] = None
    null_rejected = False
    try:
        validate_train_batch([bad] + batch[1:])
    except ValueError:
        null_rejected = True
    if not null_rejected:
        raise PilotError("tiny_gate_null_advantage_not_rejected")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONPATH": str(repo / "src")})
    log_out = run_dir / "logs" / "tiny.stdout"
    log_err = run_dir / "logs" / "tiny.stderr"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_qwen3_grpo_update_pilot.py",
    ]
    return_code = _run_logged(command, cwd=repo, env=env, stdout_path=log_out, stderr_path=log_err)
    _write_json(
        run_dir / "training" / "cpu-gates.json",
        {
            "schema_version": "t034-cpu-gates-v1",
            "status": "PASS" if return_code == 0 else "FAIL",
            "batch_validation": observed,
            "null_advantage_rejected": null_rejected,
            "tiny_threads": 2,
            "pytest_command": command,
            "exit_code": return_code,
            "stdout": str(log_out.resolve()),
            "stderr": str(log_err.resolve()),
            "created_at_utc": _now(),
        },
    )
    if return_code != 0:
        raise PilotError(f"tiny_cpu_gate_failed:{return_code}")
    return {"status": "CPU_GATES_PASS", "counts": observed}


def _make_guard(run_dir: Path, config: Any) -> BudgetGuard:
    guard = BudgetGuard(
        run_dir / "budget-state.json",
        run_id="T-034/E-031",
        run_hard_seconds=int(config.budget["run_hard_seconds"]),
        job_hard_seconds=int(config.budget["job_hard_seconds"]),
        soft_seconds=int(config.budget["soft_seconds"]),
        cleanup_reserve_seconds=int(config.budget["cleanup_reserve_seconds"]),
        gpu_count=int(config.budget["gpu_count"]),
        call_limits={key: int(value) for key, value in config.limits.items()},
    )
    return guard


def _start_guard(guard: BudgetGuard, *, recovery_elapsed: float | None = None) -> None:
    state = _read_json(guard.state_path, {}) or {}
    if state.get("active_attempt") is not None:
        if recovery_elapsed is None:
            raise PilotError("unfinished_gpu_attempt_requires_explicit_recovery_elapsed")
        guard.recover_interrupted(
            elapsed_gpu_seconds=float(recovery_elapsed), reason="T034 explicit external recovery"
        )
    if not os.environ.get("SLURM_JOB_ID"):
        raise PilotError("real_qwen3_gpu_stage_requires_slurm_job")
    guard.start_job(
        job_id=os.environ["SLURM_JOB_ID"], attempt_id=f"{os.environ['SLURM_JOB_ID']}-{os.getpid()}"
    )


def _adapter_state_fingerprint(policy: Any) -> str:
    return policy.policy_fingerprint()


def _resume_checkpoint_if_available(
    policy: Any, run_dir: Path, state: dict[str, Any]
) -> dict[str, Any]:
    if int(state.get("successful_updates", 0)) != 0:
        return state
    checkpoint = run_dir / "training" / "checkpoints" / "step-1" / "checkpoint.json"
    if not checkpoint.is_file():
        return state
    metadata = _read_json(checkpoint)
    expected_input = state.get("input_fingerprint")
    if expected_input and metadata.get("metadata", {}).get("input_fingerprint") not in {
        None,
        expected_input,
    }:
        raise PilotError("step1_checkpoint_input_hash_mismatch")
    recovered = {
        **state,
        "successful_updates": 1,
        "logical_cursor": 1,
        "resumed_from_checkpoint": True,
        "checkpoint": "training/checkpoints/step-1",
    }
    policy.load_run_state(run_dir, recovered)
    _write_json(run_dir / "training" / "state.json", recovered)
    return recovered


def _train(
    repo: Path,
    run_dir: Path,
    config_path: Path,
    *,
    recovery_elapsed: float | None = None,
) -> dict[str, Any]:
    _ensure_run_dir(run_dir)
    config, context = _config(repo, run_dir, config_path)
    _assert_parent(repo, config)
    batch_path = run_dir / "training" / "inputs" / "train-batch.jsonl"
    records = _read_jsonl(batch_path)
    counts = validate_train_batch(records)
    input_fingerprint = sha256_file(batch_path)
    state_path = run_dir / "training" / "state.json"
    state = _read_json(state_path, {}) or {}
    if state and state.get("input_fingerprint") not in {None, input_fingerprint}:
        raise PilotError("training_input_fingerprint_changed")
    state = {
        "schema_version": "t034-training-state-v1",
        "task": "T-034/E-031",
        "input_fingerprint": input_fingerprint,
        "source_batch_counts": counts,
        "successful_updates": int(state.get("successful_updates", 0)),
        "logical_cursor": int(state.get("logical_cursor", state.get("successful_updates", 0))),
        "update_records": list(state.get("update_records", [])),
        "resumed_from_checkpoint": bool(state.get("resumed_from_checkpoint", False)),
    }
    if state["successful_updates"] >= 2:
        return {"status": "TWO_LOGICAL_UPDATES_ALREADY_COMPLETE", **state}
    guard = _make_guard(run_dir, config)
    policy = None
    completed_status = "FAILED"
    try:
        _start_guard(guard, recovery_elapsed=recovery_elapsed)
        from enterprise_agent_rl.grpo.torch_backend import QwenPolicy

        policy = QwenPolicy.load_qwen3_update(context, config, guard)
        if int(policy.parent_merge.get("module_count", 0)) != EXPECTED_MODULE_COUNT:
            raise PilotError("parent_merge_module_count_mismatch")
        base_fingerprint = policy.frozen_base_fingerprint()
        initial_delta = _adapter_state_fingerprint(policy)
        initial_adapter_state = policy._adapter_state()
        if any(
            bool(value.detach().abs().max().item())
            for key, value in initial_adapter_state.items()
            if key.endswith("lora_B.weight")
        ):
            raise PilotError("initial_lora_B_not_zero")
        state["parent_base_fingerprint"] = base_fingerprint
        state["initial_delta_fingerprint"] = initial_delta
        state["initial_lora_B_zero"] = True
        state = _resume_checkpoint_if_available(policy, run_dir, state)
        for logical_step in range(int(state["successful_updates"]) + 1, 3):
            before_delta = _adapter_state_fingerprint(policy)
            result = policy.update(
                records,
                logical_step=logical_step,
                total_tokens=EXPECTED_TOTAL_TOKENS,
            )
            if result.get("status") != "UPDATED":
                raise PilotError(f"logical_step_not_updated:{logical_step}:{result}")
            after_base = policy.frozen_base_fingerprint()
            after_delta = _adapter_state_fingerprint(policy)
            if after_base != base_fingerprint:
                raise PilotError(f"base_parameters_changed:step{logical_step}")
            if after_delta == before_delta:
                raise PilotError(f"delta_parameters_did_not_change:step{logical_step}")
            if float(result.get("gradient_norm", 0.0)) <= 0 or not float(
                result.get("gradient_norm", 0.0)
            ) == float(result.get("gradient_norm", 0.0)):
                raise PilotError(f"gradient_not_finite_positive:step{logical_step}")
            metadata = {
                "logical_step": logical_step,
                "input_fingerprint": input_fingerprint,
                "source_batch_sha256": input_fingerprint,
                "parent_adapter_sha256": QWEN3_PARENT_WEIGHT_SHA256,
                "parent_adapter_config_sha256": QWEN3_PARENT_CONFIG_SHA256,
                "base_fingerprint_before": base_fingerprint,
                "base_fingerprint_after": after_base,
                "delta_fingerprint_before": before_delta,
                "delta_fingerprint_after": after_delta,
                "logical_cursor": logical_step,
            }
            checkpoint = policy.save_checkpoint(
                run_dir, logical_step=logical_step, metadata=metadata
            )
            row = {
                "logical_step": logical_step,
                "result": result,
                "checkpoint": checkpoint,
                **metadata,
            }
            state["successful_updates"] = logical_step
            state["logical_cursor"] = logical_step
            state["update_records"].append(row)
            state["checkpoint"] = checkpoint
            state["forward_backward_count"] = policy.forward_backward_count
            state["amp_skip_replays_total"] = policy.amp_skip_replays_total
            _write_json(state_path, state)
            _write_json(run_dir / "training" / f"update-step-{logical_step}.json", row)
        snapshot = policy.policy_snapshot(round_index=0, successful_updates=2)
        final = policy.freeze_final(run_dir, snapshot=snapshot)
        final["adapter_sha256"] = sha256_file(
            run_dir / "training" / "final-delta" / "adapter_model.safetensors"
        )
        final["adapter_config_sha256"] = sha256_file(
            run_dir / "training" / "final-delta" / "adapter_config.json"
        )
        final["parent_dependency"] = {
            "parent_adapter_sha256": QWEN3_PARENT_WEIGHT_SHA256,
            "parent_adapter_config_sha256": QWEN3_PARENT_CONFIG_SHA256,
            "standalone_fresh_base_load_allowed": False,
        }
        _write_json(run_dir / "training" / "final-delta.json", final)
        training_metrics = {
            "schema_version": "t034-training-metrics-v1",
            "status": "GRPO_TWO_UPDATE_PASS",
            "logical_updates": 2,
            "physical_successful_optimizer_updates": 2,
            "forward_backward": policy.forward_backward_count,
            "amp_skip_replays": policy.amp_skip_replays_total,
            "optimizer_update_attempts": _read_json(run_dir / "budget-state.json", {})
            .get("call_counters", {})
            .get("optimizer_update_attempts", 0),
            "successful_optimizer_updates": _read_json(run_dir / "budget-state.json", {})
            .get("call_counters", {})
            .get("successful_optimizer_updates", 0),
            "source_batch_sha256": input_fingerprint,
            "source_batch_counts": counts,
            "base_fingerprint_after_parent_merge": base_fingerprint,
            "base_parameters_changed": False,
            "base_gradients_present": False,
            "initial_delta_fingerprint": initial_delta,
            "initial_lora_B_zero": True,
            "final_delta_fingerprint": policy.policy_fingerprint(),
            "final_delta_sha256": final["adapter_sha256"],
            "parent_adapter_sha256": QWEN3_PARENT_WEIGHT_SHA256,
            "parent_adapter_config_sha256": QWEN3_PARENT_CONFIG_SHA256,
            "checkpoint_paths": [
                "training/checkpoints/step-1",
                "training/checkpoints/step-2",
            ],
            "extra_successful_optimizer_updates_allowed": 4,
        }
        _write_json(run_dir / "training" / "training-metrics.json", training_metrics)
        completed_status = "COMPLETED"
        return training_metrics
    finally:
        if policy is not None:
            policy.close()
        try:
            guard.close_job(status=completed_status)
        except Exception:
            if completed_status == "COMPLETED":
                raise


def _reload(
    repo: Path,
    run_dir: Path,
    config_path: Path,
    *,
    recovery_elapsed: float | None = None,
) -> dict[str, Any]:
    _ensure_run_dir(run_dir)
    final_delta = run_dir / "training" / "final-delta"
    if not (final_delta / "adapter_model.safetensors").is_file():
        raise PilotError("final_delta_missing_for_reload")
    config, context = _config(repo, run_dir, config_path, final_delta=final_delta)
    _assert_parent(repo, config)
    guard = _make_guard(run_dir, config)
    policy = None
    status = "FAILED"
    try:
        _start_guard(guard, recovery_elapsed=recovery_elapsed)
        from enterprise_agent_rl.grpo.torch_backend import QwenPolicy

        policy = QwenPolicy.load_qwen3_update_evaluation(context, config, guard, arm="RL")
        records = _read_jsonl(run_dir / "training" / "inputs" / "train-batch.jsonl")
        row = records[0]
        torch = __import__("torch")
        input_ids = torch.tensor(
            [row["prompt_token_ids"] + row["generated_token_ids"]],
            dtype=torch.long,
            device=policy.device,
        )
        attention_mask = torch.ones_like(input_ids)
        keep = len(row["generated_token_ids"]) + 1
        guard.record_call("reload_forward")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                try:
                    outputs = policy.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        num_logits_to_keep=keep,
                    )
                except TypeError:
                    outputs = policy.model(
                        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                    )
                logits = outputs.logits[:, -keep:, :]
                finite_logits = bool(torch.isfinite(logits).all().item())
        if not finite_logits:
            raise PilotError("reload_nonfinite_logits")
        identity = policy.evaluation_identity("RL")
        check = {
            "schema_version": "t034-reload-check-v1",
            "status": "RELOAD_PASS",
            "finite_logits": finite_logits,
            "forward_count": 1,
            "model_id": QWEN3_MODEL_ID,
            "revision": QWEN3_REVISION,
            "fresh_base_load_count": 1,
            "parent_merge_count": identity["parent_merge_count"],
            "delta_merge_count": identity["delta_merge_count"],
            "merge_order": ["fresh_base", "parent_sft", "final_delta"],
            "parent_adapter_sha256": QWEN3_PARENT_WEIGHT_SHA256,
            "parent_adapter_config_sha256": QWEN3_PARENT_CONFIG_SHA256,
            "adapter_sha256": sha256_file(final_delta / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256_file(final_delta / "adapter_config.json"),
            "delta_depends_on_parent": True,
            "standalone_delta_load_allowed": False,
            "evaluation_identity": identity,
        }
        _write_json(run_dir / "training" / "reload-check.json", check)
        status = "COMPLETED"
        return check
    finally:
        if policy is not None:
            policy.close()
        try:
            guard.close_job(status=status)
        except Exception:
            if status == "COMPLETED":
                raise


def _copy_validation(repo: Path, run_dir: Path) -> dict[str, Any]:
    source = (repo / T032_VALIDATION).resolve()
    manifest = source / "validation-manifest.json"
    if not manifest.is_file() or sha256_file(manifest) != T032_VALIDATION_MANIFEST_SHA256:
        raise PilotError("t032_validation_manifest_identity_mismatch")
    target = run_dir / "evaluation" / "validation"
    if target.exists():
        copied_manifest = target / "validation-manifest.json"
        if (
            not copied_manifest.is_file()
            or sha256_file(copied_manifest) != T032_VALIDATION_MANIFEST_SHA256
        ):
            raise PilotError("copied_validation_manifest_identity_mismatch")
    else:
        shutil.copytree(source, target)
    copied = target / "validation-manifest.json"
    value = _read_json(copied)
    if sha256_file(copied) != T032_VALIDATION_MANIFEST_SHA256:
        raise PilotError("copied_validation_manifest_hash_changed")
    for relative in (
        "public-panel.jsonl",
        "private/verifier-inputs.jsonl",
        "databases/validation-0.sqlite3",
        "databases/validation-1.sqlite3",
    ):
        path = target / relative
        if not path.is_file():
            raise PilotError(f"copied_validation_file_missing:{relative}")
    for path in target.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    source_info = {
        "schema_version": "t034-validation-source-v1",
        "source": str(source),
        "manifest_sha256": T032_VALIDATION_MANIFEST_SHA256,
        "copied_manifest_sha256": sha256_file(copied),
        "task_ids": value["task_ids"],
        "training_process_may_read": False,
        "test_payload_accessed": False,
        "copied_read_only": True,
    }
    _write_json(run_dir / "evaluation" / "validation-source.json", source_info)
    return source_info


def _compare_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(repo / "src"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "42",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return env


def _wait_health(
    process: subprocess.Popen[Any], endpoint: str, *, timeout_s: float = 900.0
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise PilotError(f"model_server_early_exit:{process.returncode}")
        result = subprocess.run(
            ["curl", "--noproxy", "*", "--fail", "--silent", "--show-error", endpoint],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise PilotError("model_server_health_invalid_json") from exc
            if payload.get("status") == "ready":
                return
        time.sleep(1.0)
    if process.poll() is not None:
        raise PilotError(f"model_server_early_exit:{process.returncode}")
    raise PilotError("model_server_health_deadline_exceeded")


def _stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run_evaluation_arm(
    repo: Path,
    run_dir: Path,
    arm: str,
    *,
    parent_adapter: Path | None,
    adapter: Path | None,
    port: int,
) -> dict[str, Any]:
    env = _compare_env(repo)
    engine_dir = run_dir / "evaluation" / arm / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    server_log = run_dir / "logs" / f"server-{arm}.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "enterprise_agent_rl.model_server",
        "--config",
        str((repo / MODEL_CONFIG_DEFAULT).resolve()),
        "--cache",
        str((repo / "artifacts/model-cache-qwen3-8b").resolve()),
        "--output",
        str(engine_dir.resolve()),
        "--port",
        str(port),
    ]
    if adapter is not None:
        command.extend(["--adapter", str(adapter.resolve())])
    if parent_adapter is not None:
        command.extend(["--parent-adapter", str(parent_adapter.resolve())])
    server_log_handle = server_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=env,
        stdout=server_log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_health(process, f"http://127.0.0.1:{port}/health")
        worker_command = [
            sys.executable,
            "scripts/evaluation/run_qwen3_runtime_panel.py",
            "--output",
            str(run_dir.resolve()),
            "--arm",
            arm,
            "--worker-index",
            "0",
            "--worker-count",
            "1",
            "--endpoint",
            f"http://127.0.0.1:{port}/v1/chat/completions",
            "--engine-dir",
            str(engine_dir.resolve()),
            "--model-config",
            str((repo / MODEL_CONFIG_DEFAULT).resolve()),
            "--agent-config",
            str((repo / AGENT_CONFIG_DEFAULT).resolve()),
        ]
        if adapter is not None:
            worker_command.extend(["--adapter", str(adapter.resolve())])
        if parent_adapter is not None:
            worker_command.extend(["--parent-adapter", str(parent_adapter.resolve())])
        worker_rc = _run_logged(
            worker_command,
            cwd=repo,
            env=env,
            stdout_path=run_dir / "logs" / f"worker-{arm}.stdout",
            stderr_path=run_dir / "logs" / f"worker-{arm}.stderr",
        )
        if worker_rc not in {0, 2}:
            raise PilotError(f"evaluation_worker_failed:{arm}:{worker_rc}")
        merge_command = [
            sys.executable,
            "scripts/evaluation/run_qwen3_runtime_panel.py",
            "--output",
            str(run_dir.resolve()),
            "--merge",
            "--arm",
            arm,
            "--agent-config",
            str((repo / AGENT_CONFIG_DEFAULT).resolve()),
        ]
        merge_rc = _run_logged(
            merge_command,
            cwd=repo,
            env=env,
            stdout_path=run_dir / "logs" / f"merge-{arm}.stdout",
            stderr_path=run_dir / "logs" / f"merge-{arm}.stderr",
        )
        if merge_rc not in {0, 2}:
            raise PilotError(f"evaluation_merge_failed:{arm}:{merge_rc}")
    finally:
        _stop_server(process)
        server_log_handle.close()
    metrics = _read_json(run_dir / "evaluation" / arm / "metrics.json")
    if not isinstance(metrics, dict):
        raise PilotError(f"evaluation_metrics_missing:{arm}")
    return metrics


def _comparison(run_dir: Path, validation: dict[str, Any]) -> dict[str, Any]:
    sft = _read_json(run_dir / "evaluation" / "sft" / "metrics.json")
    rl = _read_json(run_dir / "evaluation" / "rl" / "metrics.json")
    if not isinstance(sft, dict) or not isinstance(rl, dict):
        raise PilotError("evaluation_arm_metrics_missing")
    if sft.get("task_ids") != rl.get("task_ids") or sft.get("task_ids") != validation["task_ids"]:
        raise PilotError("evaluation_task_set_mismatch")
    sft_records = {
        row["task_id"]: row
        for row in _read_jsonl(run_dir / "evaluation" / "sft" / "case-results.jsonl")
    }
    rl_records = {
        row["task_id"]: row
        for row in _read_jsonl(run_dir / "evaluation" / "rl" / "case-results.jsonl")
    }
    transitions = []
    for task_id in validation["task_ids"]:
        transitions.append(
            {
                "task_id": task_id,
                "sft_success": bool(sft_records[task_id]["success"]),
                "rl_success": bool(rl_records[task_id]["success"]),
                "sft_category": sft_records[task_id]["category"],
                "rl_category": rl_records[task_id]["category"],
                "sft_terminal_reason": sft_records[task_id]["terminal_reason"],
                "rl_terminal_reason": rl_records[task_id]["terminal_reason"],
            }
        )
    complete = (
        sft.get("status") == "FROZEN_COMPARISON_ARM_COMPLETE"
        and rl.get("status") == "FROZEN_COMPARISON_ARM_COMPLETE"
        and sft.get("counts", {}).get("planned") == 20
        and rl.get("counts", {}).get("planned") == 20
        and sft.get("counts", {}).get("unknown") == 0
        and rl.get("counts", {}).get("unknown") == 0
    )
    sft_successes = int(sft["counts"]["successes"])
    rl_successes = int(rl["counts"]["successes"])
    comparison = {
        "schema_version": "t034-comparison-v1",
        "status": "FROZEN_COMPARISON_COMPLETE" if complete else "FROZEN_COMPARISON_INCOMPLETE",
        "task": "T-034/E-031",
        "task_ids": validation["task_ids"],
        "planned_denominator": 20,
        "sft": {
            "status": sft["status"],
            "successes": sft_successes,
            "success_rate": sft.get("business_success_rate"),
            "scored": sft["counts"].get("scored"),
            "unknown": sft["counts"].get("unknown"),
        },
        "rl": {
            "status": rl["status"],
            "successes": rl_successes,
            "success_rate": rl.get("business_success_rate"),
            "scored": rl["counts"].get("scored"),
            "unknown": rl["counts"].get("unknown"),
        },
        "primary_endpoint": {
            "rl_successes_minus_sft_successes": rl_successes - sft_successes if complete else None,
            "rl_success_rate_minus_sft_success_rate": (
                rl["business_success_rate"] - sft["business_success_rate"] if complete else None
            ),
            "fixed_denominator": 20,
        },
        "transitions": transitions,
        "model_identity": {
            "model_id": QWEN3_MODEL_ID,
            "revision": QWEN3_REVISION,
            "sft_parent_adapter_sha256": QWEN3_PARENT_WEIGHT_SHA256,
            "rl_parent_adapter_sha256": QWEN3_PARENT_WEIGHT_SHA256,
            "rl_delta_path": str((run_dir / "training" / "final-delta").resolve()),
            "rl_delta_sha256": _read_json(run_dir / "training" / "final-delta.json")[
                "adapter_sha256"
            ],
            "sft_load_order": ["fresh_base", "parent_sft_once"],
            "rl_load_order": ["fresh_base", "parent_sft_once", "delta_once"],
            "base_third_arm_run": False,
        },
        "resource_limits": {
            "model_requests": 1600,
            "runtime_tool_attempts": 600,
            "verifier_calls": 48,
            "verifier_sql": 96,
        },
        "resource_observed": {
            "model_requests": int(sft.get("model_physical_requests", 0))
            + int(rl.get("model_physical_requests", 0)),
            "runtime_tool_attempts": int(sft.get("tool_attempts", 0))
            + int(rl.get("tool_attempts", 0)),
            "verifier_calls": int(sft.get("verifier_calls", 0)) + int(rl.get("verifier_calls", 0)),
            "verifier_sql": int(sft.get("verifier_sql_calls", 0))
            + int(rl.get("verifier_sql_calls", 0)),
        },
        "validation_manifest_sha256": T032_VALIDATION_MANIFEST_SHA256,
        "finished_at_utc": _now(),
    }
    for key, limit in comparison["resource_limits"].items():
        if comparison["resource_observed"][key] > limit:
            raise PilotError(f"evaluation_call_limit_exceeded:{key}")
    _write_json(run_dir / "metrics" / "comparison.json", comparison)
    _write_json(run_dir / "metrics" / "summary.json", comparison)
    _write_json(
        run_dir / "metrics.json",
        {
            "schema_version": "t034-metrics-v1",
            "status": comparison["status"],
            "comparison": "metrics/comparison.json",
            "sft": sft,
            "rl": rl,
        },
    )
    return comparison


def _compare(repo: Path, run_dir: Path, config_path: Path) -> dict[str, Any]:
    _ensure_run_dir(run_dir)
    if not (run_dir / "training" / "reload-check.json").is_file():
        raise PilotError("reload_check_required_before_comparison")
    validation = _copy_validation(repo, run_dir)
    parent = _resolved(
        str(_read_json(run_dir / "frozen-config.json")["config"]["model"]["parent_adapter"]), repo
    )
    delta = run_dir / "training" / "final-delta"
    sft = _run_evaluation_arm(repo, run_dir, "sft", parent_adapter=None, adapter=parent, port=18081)
    rl = _run_evaluation_arm(repo, run_dir, "rl", parent_adapter=parent, adapter=delta, port=18082)
    comparison = _comparison(run_dir, validation)
    _write_json(
        run_dir / "evaluation" / "comparison-resource-summary.json",
        {
            "sft": sft,
            "rl": rl,
            "validation_manifest_sha256": T032_VALIDATION_MANIFEST_SHA256,
            "base_third_arm_run": False,
        },
    )
    return comparison


def _record_allocation(
    repo: Path,
    run_dir: Path,
    *,
    stage: str,
    status: str,
    exit_code: int,
    elapsed_s: float,
    gpu_count: int,
    slurm_evidence: Path | None,
) -> dict[str, Any]:
    _ensure_run_dir(run_dir)
    ledger = _read_json(run_dir / "allocation-ledger.json", {}) or {}
    records = list(ledger.get("records", []))
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise PilotError("allocation_record_requires_slurm_job")
    key = f"{job_id}:{stage}"
    if any(record.get("key") == key for record in records):
        raise PilotError(f"allocation_record_duplicate:{key}")
    wrapper_gpu_seconds = float(elapsed_s) * int(gpu_count)
    record = {
        "key": key,
        "stage": stage,
        "status": status,
        "exit_code": int(exit_code),
        "job_id": job_id,
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
        "gpu_count": int(gpu_count),
        "wrapper_elapsed_s": float(elapsed_s),
        "wrapper_gpu_seconds": wrapper_gpu_seconds,
        "wrapper_gpu_minutes": wrapper_gpu_seconds / 60.0,
        "slurm_accounting_evidence_path": str(slurm_evidence.resolve()) if slurm_evidence else None,
        "recorded_at_utc": _now(),
    }
    records.append(record)
    total = sum(float(item.get("wrapper_gpu_seconds", 0.0)) for item in records)
    ledger = {
        "schema_version": "t034-allocation-ledger-v1",
        "task": "T-034/E-031",
        "records": records,
        "total_gpu_seconds": total,
        "total_gpu_minutes": total / 60.0,
        "budget_gpu_seconds": 10800,
        "old_t033_gpu_minutes_excluded": 15.633333333333334,
        "budget_within_limit": total <= 10800 + 1e-6,
    }
    _write_json(run_dir / "allocation-ledger.json", ledger)
    chain = _read_json(run_dir / "job-chain.json", {}) or {}
    jobs = list(chain.get("jobs", []))
    jobs.append(record)
    chain["jobs"] = jobs
    chain["last_job"] = record
    _write_json(run_dir / "job-chain.json", chain)
    if total > 10800 + 1e-6:
        raise PilotError("t034_gpu_budget_exceeded")
    return record


def _artifact_manifest(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact-manifest.json"}
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative in excluded or relative.endswith(".lock"):
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "t034-artifact-manifest-v1",
        "task": "T-034/E-031",
        "scope": "all T-034 run files except this mutable manifest and BudgetGuard lock files",
        "excluded_paths": ["artifact-manifest.json", "**/*.lock"],
        "files": files,
        "file_count": len(files),
        "created_at_utc": _now(),
    }
    _write_json(run_dir / "artifact-manifest.json", manifest)
    return manifest


def _finalize(repo: Path, run_dir: Path, config_path: Path) -> dict[str, Any]:
    _ensure_run_dir(run_dir)
    training = _read_json(run_dir / "training" / "training-metrics.json", {}) or {}
    reload_check = _read_json(run_dir / "training" / "reload-check.json", {}) or {}
    comparison = _read_json(run_dir / "metrics" / "comparison.json", {}) or {}
    allocation = _read_json(run_dir / "allocation-ledger.json", {}) or {}
    cpu = _read_json(run_dir / "training" / "cpu-gates.json", {}) or {}
    grpo_pass = (
        training.get("status") == "GRPO_TWO_UPDATE_PASS"
        and reload_check.get("status") == "RELOAD_PASS"
    )
    comparison_pass = comparison.get("status") == "FROZEN_COMPARISON_COMPLETE"
    overall = "T034_COMPLETE" if grpo_pass and comparison_pass else "T034_INCOMPLETE"
    batch_counts = training.get("source_batch_counts", {})
    logical_updates = training.get("logical_updates")
    physical_steps = training.get("physical_successful_optimizer_updates")
    forward_backward = training.get("forward_backward")
    amp_replays = training.get("amp_skip_replays")
    sft_summary = comparison.get("sft", {})
    rl_summary = comparison.get("rl", {})
    endpoint = comparison.get("primary_endpoint", {})
    report = [
        "# T-034 / E-031 — 固定 Qwen3 两次 GRPO 更新与 SFT/RL 冻结对照",
        "",
        f"- 总状态：`{overall}`；GRPO/reload=`{grpo_pass}`；comparison=`{comparison_pass}`。",
        f"- 绝对产物路径：`{run_dir.resolve()}`。",
        "",
        "## GRPO 两次更新",
        "",
        f"- 逻辑更新：`{logical_updates}`；物理成功 optimizer.step：`{physical_steps}`；",
        f"forward/backward=`{forward_backward}`，AMP replay=`{amp_replays}`。",
        f"- 固定分母：records=`{batch_counts.get('records')}`，",
        f"active=`{batch_counts.get('active_records')}`，",
        f"zero=`{batch_counts.get('zero_records')}`，tokens=`{batch_counts.get('total_tokens')}`，",
        f"active tokens=`{batch_counts.get('active_tokens')}`，",
        f"max sequence=`{batch_counts.get('max_full_sequence_tokens')}`。",
        f"- 父 Adapter SHA：`{training.get('parent_adapter_sha256')}`；",
        f"final delta SHA：`{training.get('final_delta_sha256')}`；",
        "delta 只允许在父 SFT merge 后加载，不能单独套在原始 Base 上。",
        "",
        "## 独立重载",
        "",
        f"- `{reload_check.get('status')}`；",
        f"fresh Base load=`{reload_check.get('fresh_base_load_count')}`，",
        f"parent merge=`{reload_check.get('parent_merge_count')}`，",
        f"delta merge=`{reload_check.get('delta_merge_count')}`，",
        f"finite logits=`{reload_check.get('finite_logits')}`。",
        "",
        "## 冻结 SFT/RL 对照",
        "",
    ]
    if comparison:
        report.extend(
            [
                f"- SFT：`{sft_summary.get('successes')}/20`，",
                f"scored=`{sft_summary.get('scored')}`，unknown=`{sft_summary.get('unknown')}`。",
                f"- RL：`{rl_summary.get('successes')}/20`，",
                f"scored=`{rl_summary.get('scored')}`，unknown=`{rl_summary.get('unknown')}`。",
                f"- 主终点 RL−SFT successes=`{endpoint.get('rl_successes_minus_sft_successes')}`，",
                f"rate diff=`{endpoint.get('rl_success_rate_minus_sft_success_rate')}`，",
                "固定分母 20。",
                "- 逐任务转移见 `metrics/comparison.json`；不运行 Base 第三臂。",
            ]
        )
    else:
        report.append("- comparison 尚未完成；主终点保持 null，未缩小分母。")
    report.extend(
        [
            "",
            "## 资源、边界与修复",
            "",
            f"- 新 T-034 allocation：`{allocation.get('total_gpu_seconds', 0.0)}` GPU秒 /",
            f"`{allocation.get('total_gpu_minutes', 0.0)}` GPU分钟；预算上限 10800 GPU秒。",
            "旧 T-033 的 15.6333 GPU分钟单列且未混入；Job 链见 `job-chain.json`。",
            f"- CPU gate：`{cpu.get('status')}`；T-033 prepare 已改为 append-only 控制账本。",
            "历史原包实际 28 次 reference SQL 保留，不改写成 4。",
            "- 训练进程未读取 validation/test payload；评测复制 T-032 manifest，",
            "manifest SHA 保持固定值，DB/private/public 只读。",
            "- 业务失败、普通 Action/SQL 错误与 budget exhaustion 保留为业务结果；",
            "不以零/负收益追改模型或步数。",
            "",
            "## 证据与 Git",
            "",
            "- `training/` 保存 batch、advantage 审计、step-1/step-2 checkpoint、",
            "AMP/optimizer 状态、final delta 和 reload identity。",
            "- `evaluation/sft/` 与 `evaluation/rl/` 保存请求、响应、engine、",
            "SQL audit、trajectory、Verifier、case-results；原始 stdout/stderr 在 `logs/`。",
            "- `artifact-manifest.json` 排除自身与 BudgetGuard lock；代码快照和 dirty patch 在",
            "`source-snapshot/`，不包含受保护路径。",
            "",
            "固定批次两次更新及本次 SFT/RL 比较后停止；后续由总控依据实际结果规划。",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    checks = {
        "schema_version": "t034-checks-v1",
        "grpo_two_update_reload": "PASS" if grpo_pass else "INCOMPLETE",
        "frozen_comparison": "PASS" if comparison_pass else "INCOMPLETE",
        "cpu_gates": cpu.get("status"),
        "allocation_within_10800_gpu_seconds": allocation.get("budget_within_limit"),
        "primary_endpoint": comparison.get("primary_endpoint") if comparison else None,
        "unfinished_next_action": (
            None
            if overall == "T034_COMPLETE"
            else "inspect persisted stage evidence; do not resample"
        ),
    }
    _write_json(run_dir / "checks.json", checks)
    manifest = _artifact_manifest(run_dir)
    return {"status": overall, "checks": checks, "artifact_file_count": manifest["file_count"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "tiny", "train", "reload", "compare", "record-allocation", "finalize"),
        required=True,
    )
    parser.add_argument("--output", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--recovery-elapsed-gpu-seconds", type=float)
    parser.add_argument("--record-status", default="COMPLETED")
    parser.add_argument("--record-exit-code", type=int, default=0)
    parser.add_argument("--record-elapsed-s", type=float, default=0.0)
    parser.add_argument("--record-gpu-count", type=int, default=1)
    parser.add_argument("--record-slurm-evidence", type=Path)
    args = parser.parse_args()
    repo = args.repo_root.expanduser().resolve()
    run_dir = args.output.expanduser()
    if not run_dir.is_absolute():
        run_dir = repo / run_dir
    run_dir = run_dir.resolve()
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = repo / config_path
    config_path = config_path.resolve()
    try:
        if args.stage == "prepare":
            result = _prepare(repo, run_dir, config_path)
        elif args.stage == "tiny":
            result = _tiny(repo, run_dir, config_path)
        elif args.stage == "train":
            result = _train(
                repo,
                run_dir,
                config_path,
                recovery_elapsed=args.recovery_elapsed_gpu_seconds,
            )
        elif args.stage == "reload":
            result = _reload(
                repo,
                run_dir,
                config_path,
                recovery_elapsed=args.recovery_elapsed_gpu_seconds,
            )
        elif args.stage == "compare":
            result = _compare(repo, run_dir, config_path)
        elif args.stage == "record-allocation":
            result = _record_allocation(
                repo,
                run_dir,
                stage=os.environ.get("T034_STAGE", "unknown"),
                status=args.record_status,
                exit_code=args.record_exit_code,
                elapsed_s=args.record_elapsed_s,
                gpu_count=args.record_gpu_count,
                slurm_evidence=args.record_slurm_evidence,
            )
        else:
            result = _finalize(repo, run_dir, config_path)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (PilotError, BudgetError, OSError, ValueError) as exc:
        print(f"T034_ERROR:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
