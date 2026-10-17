#!/usr/bin/env python3
"""Teacher-forced SFT behavior attribution diagnostic.

This module has two deliberately separate execution paths:

* ``preflight`` reads only the frozen train JSONL/metadata, local model cache,
  checkpoint metadata, and tokenizer.  It never imports a model class or loads
  model weights.
* ``gpu`` runs one first-generation request for every target at each of the
  four frozen stages.  It reuses :class:`QwenEngine`, the fixed tokenizer/chat
  template, the runtime ``parse_action`` parser, and the manual LoRA merge.

The GPU path never imports the benchmark manifest, environment, verifier,
reward, SQL worker, or reference teacher.  Its only score is canonical Action
comparison against the current train JSONL assistant message.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_agent_rl.model_protocol import GenerationSettings, parse_action
from enterprise_agent_rl.sft_training import load_dataset_rows, sha256_file

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET_SHA256 = "e3a583be6fd7b399e9088f098f317d7589bf564dcce506b0d8a5de825ea40e3c"
ACTION_KINDS = ("plan", "schema", "dictionary", "sql", "check", "finish")
TOOL_KINDS = frozenset(("schema", "dictionary", "sql", "check"))
PARSE_MODE = "json_fence"
MAX_CONTEXT_TOKENS = 8192


@dataclass(frozen=True)
class StageSpec:
    name: str
    checkpoint_stage: str
    global_step: int
    adapter_relative: str | None


STAGES = (
    StageSpec("base", "base", 0, None),
    StageSpec(
        "epoch-1",
        "epoch-1",
        5,
        "artifacts/sft-pilot-20390015/checkpoints/checkpoint-epoch-1/adapter",
    ),
    StageSpec(
        "epoch-2",
        "epoch-2",
        10,
        "artifacts/sft-pilot-20390015/checkpoints/checkpoint-epoch-2/adapter",
    ),
    StageSpec(
        "epoch-3",
        "epoch-3",
        15,
        "artifacts/sft-pilot-20390015/checkpoints/checkpoint-epoch-3/adapter",
    ),
)


@dataclass(frozen=True)
class Target:
    dataset_row_index: int
    dataset_line_number: int
    example_id: str
    task_id: str
    teacher_source: str
    template_id: str
    assistant_message_index: int
    history: tuple[dict[str, str], ...]
    history_sha256: str
    raw_gold_action: str
    gold_action: dict[str, Any]
    gold_kind: str

    @property
    def is_tool_target(self) -> bool:
        return self.gold_kind in TOOL_KINDS

    @property
    def target_id(self) -> str:
        return f"{self.example_id}:assistant-{self.assistant_message_index}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def write_jsonl_record(stream: Any, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def command_result(argv: list[str], timeout: float = 30.0) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # pragma: no cover - host-dependent
        return {
            "argv": argv,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def git_state() -> dict[str, Any]:
    status = command_result(["git", "status", "--short"])
    commit = command_result(["git", "rev-parse", "HEAD"])
    branch = command_result(["git", "branch", "--show-current"])
    status_text = status.get("stdout", "")
    return {
        "commit": commit.get("stdout", "").strip() or None,
        "branch": branch.get("stdout", "").strip() or None,
        "status_short": status_text,
        "dirty": bool(status_text),
        "untracked": [line[3:] for line in status_text.splitlines() if line.startswith("?? ")],
        "status_command": status,
    }


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def build_targets(rows: list[dict[str, Any]]) -> list[Target]:
    """Rebuild every target from ``messages[:assistant_index]`` only."""

    targets: list[Target] = []
    for row_index, row in enumerate(rows):
        messages = row["messages"]
        indices = row["assistant_message_indices"]
        previous_index = -1
        for assistant_index in indices:
            if assistant_index <= previous_index:
                raise ValueError(f"assistant_indices_not_strictly_increasing:{row['task_id']}")
            previous_index = assistant_index
            if assistant_index >= len(messages):
                raise ValueError(f"assistant_index_out_of_range:{row['task_id']}")
            target_message = messages[assistant_index]
            if target_message.get("role") != "assistant":
                raise ValueError(f"target_not_assistant:{row['task_id']}:{assistant_index}")
            if assistant_index == 0 or messages[assistant_index - 1].get("role") != "user":
                raise ValueError(f"target_has_no_user_prompt:{row['task_id']}:{assistant_index}")
            history = messages[:assistant_index]
            if target_message["content"] in [message.get("content") for message in history]:
                raise ValueError(
                    f"current_gold_action_repeated_in_history:{row['task_id']}:{assistant_index}"
                )
            gold, _ = parse_action(target_message["content"], PARSE_MODE)
            gold_dump = gold.model_dump()
            source = row["source"]
            history_copy = tuple(
                {"role": str(message["role"]), "content": str(message["content"])}
                for message in history
            )
            targets.append(
                Target(
                    dataset_row_index=row_index,
                    dataset_line_number=row_index + 1,
                    example_id=str(row["example_id"]),
                    task_id=str(row["task_id"]),
                    teacher_source=str(source["trajectory_source"]),
                    template_id=str(source["template_id"]),
                    assistant_message_index=assistant_index,
                    history=history_copy,
                    history_sha256=sha256_json(list(history_copy)),
                    raw_gold_action=str(target_message["content"]),
                    gold_action=gold_dump,
                    gold_kind=str(gold.kind),
                )
            )
    return targets


def target_fingerprint(targets: Iterable[Target]) -> str:
    values = [
        {
            "target_id": target.target_id,
            "dataset_row_index": target.dataset_row_index,
            "assistant_message_index": target.assistant_message_index,
            "teacher_source": target.teacher_source,
            "template_id": target.template_id,
            "history_sha256": target.history_sha256,
            "gold_action": target.gold_action,
        }
        for target in targets
    ]
    return sha256_json(values)


def _snapshot_path(cache: Path, revision: str) -> Path:
    return cache / "models--Qwen--Qwen2.5-1.5B-Instruct" / "snapshots" / revision


def validate_cache(cache: Path, config: Any) -> dict[str, Any]:
    manifest_path = cache / "qwen-cache-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("cache_manifest_missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("cache_manifest_model_or_revision_mismatch")
    if manifest.get("weight_files") != ["model.safetensors"]:
        raise ValueError("cache_manifest_weight_files_mismatch")
    snapshot = _snapshot_path(cache, MODEL_REVISION)
    manifest_snapshot = Path(str(manifest.get("snapshot", "")))
    if (
        not snapshot.is_dir()
        or manifest_snapshot.name != MODEL_REVISION
        or manifest_snapshot.parent.name != "snapshots"
    ):
        raise ValueError("cache_snapshot_path_mismatch")
    incomplete = sorted(
        path.name
        for path in (cache / "models--Qwen--Qwen2.5-1.5B-Instruct" / "blobs").glob("*.incomplete")
    )
    if incomplete:
        raise ValueError(f"cache_incomplete_blobs:{incomplete}")

    records: dict[str, Any] = {}
    for name, expected in sorted(manifest.get("files", {}).items()):
        path = snapshot / name
        if not path.is_file():
            raise ValueError(f"cache_file_missing:{name}")
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        record = {
            "path": str(path),
            "bytes": actual_bytes,
            "sha256": actual_sha256,
            "manifest_bytes": expected.get("bytes"),
            "manifest_sha256": expected.get("sha256"),
            "matches_manifest": actual_bytes == expected.get("bytes")
            and actual_sha256 == expected.get("sha256"),
        }
        if not record["matches_manifest"]:
            raise ValueError(f"cache_file_hash_mismatch:{name}")
        records[name] = record
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "snapshot": str(snapshot),
        "incomplete_blobs": incomplete,
        "files": records,
        "manifest": manifest,
        "config_model_id": config.model_id,
        "config_revision": config.revision,
    }


def _checkpoint_required_files() -> tuple[str, ...]:
    return (
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "adapter/added_tokens.json",
        "adapter/merges.txt",
        "adapter/special_tokens_map.json",
        "adapter/tokenizer.json",
        "adapter/tokenizer_config.json",
        "adapter/vocab.json",
        "optimizer.pt",
        "rng_state.pth",
        "scaler.pt",
        "scheduler.pt",
        "trainer_state.json",
    )


def validate_checkpoints(root: Path, config: Any) -> dict[str, Any]:
    checkpoints: dict[str, Any] = {}
    expected_steps = {"epoch-1": 5, "epoch-2": 10, "epoch-3": 15}
    for stage_name, expected_step in expected_steps.items():
        checkpoint = (
            root / "artifacts" / "sft-pilot-20390015" / "checkpoints" / (f"checkpoint-{stage_name}")
        )
        if not checkpoint.is_dir():
            raise ValueError(f"checkpoint_directory_missing:{checkpoint}")
        missing = [
            name for name in _checkpoint_required_files() if not (checkpoint / name).is_file()
        ]
        if missing:
            raise ValueError(f"checkpoint_files_missing:{stage_name}:{missing}")
        trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
        if trainer_state.get("global_step") != expected_step:
            raise ValueError(f"checkpoint_global_step_mismatch:{stage_name}")
        adapter = checkpoint / "adapter"
        adapter_config = json.loads((adapter / "adapter_config.json").read_text())
        expected_config = {
            "base_model_name_or_path": MODEL_ID,
            "base_model_revision": MODEL_REVISION,
            "implementation": "enterprise_agent_rl.manual_lora_v1",
            "r": 8,
            "lora_alpha": 16.0,
        }
        for key, expected in expected_config.items():
            if adapter_config.get(key) != expected:
                raise ValueError(f"adapter_config_mismatch:{stage_name}:{key}")
        weight = adapter / "adapter_model.safetensors"
        checkpoints[stage_name] = {
            "checkpoint_path": str(checkpoint),
            "adapter_path": str(adapter),
            "global_step": expected_step,
            "trainer_state": {
                "global_step": trainer_state.get("global_step"),
                "epoch": trainer_state.get("epoch"),
            },
            "adapter_config": adapter_config,
            "adapter_weight_path": str(weight),
            "adapter_weight_bytes": weight.stat().st_size,
            "adapter_weight_sha256": sha256_file(weight),
            "required_files": {
                name: {
                    "bytes": (checkpoint / name).stat().st_size,
                    "sha256": sha256_file(checkpoint / name),
                }
                for name in _checkpoint_required_files()
            },
            "model_id": config.model_id,
            "revision": config.revision,
        }
    return checkpoints


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise ValueError("tokenizer_returned_non_list")
    return [int(item) for item in value]


def tokenizer_prompt_tokens(tokenizer: Any, history: list[dict[str, str]]) -> int:
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(text)
    return len(_token_ids(encoded["input_ids"]))


def load_tokenizer_evidence(cache: Path, config: Any, targets: list[Target]) -> dict[str, Any]:
    # This is intentionally the only model-related import in the CPU path.
    # AutoTokenizer loads tokenizer files, never model weights.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=str(cache),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    chat_template = tokenizer.chat_template
    if not chat_template:
        raise ValueError("chat_template_missing")
    prompt_records: list[dict[str, Any]] = []
    for target in targets:
        prompt_tokens = tokenizer_prompt_tokens(tokenizer, list(target.history))
        if prompt_tokens + 512 > config.max_context_tokens:
            raise ValueError(f"context_budget_exceeded:{target.target_id}:{prompt_tokens}")
        prompt_records.append(
            {
                "target_id": target.target_id,
                "history_sha256": target.history_sha256,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": 512,
                "context_tokens": prompt_tokens + 512,
            }
        )
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "chat_template_sha256": sha256_bytes(canonical_json(chat_template).encode("utf-8")),
        "chat_template_bytes": len(canonical_json(chat_template).encode("utf-8")),
        "prompt_records": prompt_records,
        "model_weights_loaded": False,
    }


def parser_evidence() -> dict[str, Any]:
    import enterprise_agent_rl.model_protocol as protocol

    source_path = Path(protocol.__file__).resolve()
    return {
        "module": "enterprise_agent_rl.model_protocol",
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "parse_action_source_sha256": sha256_bytes(
            re.sub(r"\s+", " ", inspect.getsource(parse_action)).encode("utf-8")
        ),
        "parse_mode": PARSE_MODE,
        "action_kinds": list(ACTION_KINDS),
    }


def fixed_generation_evidence(config: Any) -> dict[str, Any]:
    settings = GenerationSettings(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        format_retries=0,
        parse_mode=PARSE_MODE,
        seed=17,
    )
    return {
        "requested": settings.model_dump(),
        "effective_transformers_generation": {
            "do_sample": False,
            "max_new_tokens": settings.max_tokens,
            "use_cache": True,
            "pad_token_id": "tokenizer.eos_token_id",
            "temperature_applied": False,
            "top_p_applied": False,
            "top_k_applied": False,
            "seed": settings.seed,
            "attention_backend": config.attention_backend,
        },
        "format_retry_policy": {
            "format_retries": 0,
            "first_generation_only": True,
        },
        "config_generation_for_reference_only": config.generation.model_dump(),
    }


def validate_rows_and_targets(
    dataset: Path, metadata_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Target], dict[str, Any]]:
    if sha256_file(dataset) != DATASET_SHA256:
        raise ValueError("dataset_sha256_mismatch")
    rows, metadata = load_dataset_rows(dataset, metadata_path)
    if metadata.get("source_split") != "train":
        raise ValueError("dataset_not_train_split")
    if metadata.get("validation_used_for_training") is not False:
        raise ValueError("dataset_validation_flag_not_false")
    if metadata.get("test_used_for_training") is not False:
        raise ValueError("dataset_test_flag_not_false")
    if any(_contains_key(row, "reference_sql") for row in rows):
        raise ValueError("reference_sql_leakage_in_dataset_row")
    targets = build_targets(rows)
    if len(rows) != 20 or len(targets) != 108:
        raise ValueError(f"dataset_target_count_mismatch:{len(rows)}:{len(targets)}")
    source_counts = Counter(target.teacher_source for target in targets)
    row_source_counts = Counter(row["source"]["trajectory_source"] for row in rows)
    if dict(row_source_counts) != {
        "deterministic_fixture_expert": 6,
        "reference_sql_teacher": 14,
    }:
        raise ValueError(f"teacher_row_source_distribution:{dict(row_source_counts)}")
    if len({target.template_id for target in targets}) != 10:
        raise ValueError("template_count_mismatch")
    tool_count = sum(target.is_tool_target for target in targets)
    if tool_count != 68:
        raise ValueError(f"tool_target_count_mismatch:{tool_count}")
    checks = {
        "row_count": len(rows),
        "target_count": len(targets),
        "tool_target_count": tool_count,
        "gold_kind_counts": dict(sorted(Counter(target.gold_kind for target in targets).items())),
        "gold_teacher_source_target_counts": dict(sorted(source_counts.items())),
        "teacher_source_row_counts": dict(sorted(row_source_counts.items())),
        "template_target_counts": dict(
            sorted(Counter(target.template_id for target in targets).items())
        ),
        "target_fingerprint": target_fingerprint(targets),
        "gold_history": {
            "current_target_index_excluded": True,
            "raw_current_gold_action_repeated_in_prior_history": False,
            "history_reconstructed_from_messages_prefix": True,
        },
    }
    return rows, metadata, targets, checks


def build_preflight_report(
    *,
    root: Path,
    dataset: Path,
    metadata_path: Path,
    cache: Path,
    model_config: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from enterprise_agent_rl.model_server import ModelSettings

    config = ModelSettings.load(model_config)
    if config.model_id != MODEL_ID or config.revision != MODEL_REVISION:
        raise ValueError("model_config_identity_mismatch")
    if config.dtype != "float16" or config.attention_backend != "sdpa_math":
        raise ValueError("model_config_runtime_mismatch")
    if config.max_context_tokens != MAX_CONTEXT_TOKENS:
        raise ValueError("model_config_context_mismatch")
    if config.generation.parse_mode != PARSE_MODE:
        raise ValueError("model_config_parser_mode_mismatch")

    _rows, metadata, targets, data_checks = validate_rows_and_targets(dataset, metadata_path)
    cache_evidence = validate_cache(cache, config)
    checkpoint_evidence = validate_checkpoints(root, config)
    tokenizer_evidence = load_tokenizer_evidence(cache, config, targets)
    generation = fixed_generation_evidence(config)
    parser = parser_evidence()
    git = git_state()

    preflight = {
        "schema_version": "1.0",
        "status": "READY_FOR_GPU_SUBMISSION",
        "run_id": run_dir.name,
        "authorization_scope": {
            "diagnostic_only": True,
            "training_allowed": False,
            "grpo_allowed": False,
            "formal_benchmark_rerun_allowed": False,
            "validation_or_test_manifest_loaded": False,
            "validation_or_test_task_executed": False,
            "environment_verifier_reward_used_for_scoring": False,
        },
        "data_boundary": {
            "dataset_path": str(dataset),
            "metadata_path": str(metadata_path),
            "dataset_sha256": sha256_file(dataset),
            "metadata_sha256": sha256_file(metadata_path),
            "source_split": metadata["source_split"],
            "validation_used_for_training": metadata["validation_used_for_training"],
            "test_used_for_training": metadata["test_used_for_training"],
            "reference_sql_key_found": False,
            "manifest_loader_called": False,
        },
        "data_checks": data_checks,
        "model_config": {
            "path": str(model_config),
            "sha256": sha256_file(model_config),
            "model_id": config.model_id,
            "revision": config.revision,
            "dtype": config.dtype,
            "attention_backend": config.attention_backend,
            "max_context_tokens": config.max_context_tokens,
        },
        "cache": cache_evidence,
        "checkpoints": checkpoint_evidence,
        "tokenizer": tokenizer_evidence,
        "generation": generation,
        "parser": parser,
        "stages": [
            {
                "name": stage.name,
                "checkpoint_stage": stage.checkpoint_stage,
                "global_step": stage.global_step,
                "adapter_path": stage.adapter_relative,
                "adapter_weight_sha256": (
                    None
                    if stage.adapter_relative is None
                    else checkpoint_evidence[stage.checkpoint_stage]["adapter_weight_sha256"]
                ),
            }
            for stage in STAGES
        ],
        "cpu_preflight": {
            "model_weights_loaded": False,
            "gpu_used": False,
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "git": git,
    }
    environment = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "phase": "cpu_preflight_complete_waiting_gpu",
        "created_at_epoch_s": time.time(),
        "cpu": preflight["cpu_preflight"],
        "python_executable": sys.executable,
        "torch_weights_loaded": False,
        "gpu_execution": None,
        "git_at_preflight": git,
        "input_fingerprints": {
            "dataset_sha256": preflight["data_boundary"]["dataset_sha256"],
            "metadata_sha256": preflight["data_boundary"]["metadata_sha256"],
            "cache_manifest_sha256": cache_evidence["manifest_sha256"],
            "target_fingerprint": data_checks["target_fingerprint"],
            "tokenizer_chat_template_sha256": tokenizer_evidence["chat_template_sha256"],
            "parser_source_sha256": parser["source_sha256"],
        },
    }
    return preflight, environment


def classify_output(
    *,
    raw_output: Any,
    response_error: dict[str, Any] | None,
    finish_reason: str | None,
) -> dict[str, Any]:
    raw = raw_output if isinstance(raw_output, str) else None
    raw_json_valid = False
    strict_envelope_valid = False
    raw_json_value: Any = None
    if raw is not None and raw.strip():
        try:
            raw_json_value = json.loads(raw)
            raw_json_valid = True
        except (json.JSONDecodeError, RecursionError):
            raw_json_valid = False
        if isinstance(raw_json_value, dict):
            strict_envelope_valid = (
                set(raw_json_value) == {"kind", "arguments"}
                and raw_json_value.get("kind") in ACTION_KINDS
                and isinstance(raw_json_value.get("arguments"), dict)
            )

    parsed_action: dict[str, Any] | None = None
    parser_error: str | None = None
    parser_normalized = False
    if raw is None or not raw.strip():
        parser_error = (
            str(response_error.get("code"))
            if response_error and response_error.get("code")
            else "empty_output"
        )
    else:
        try:
            action, parser_normalized = parse_action(raw, PARSE_MODE)
            parsed_action = action.model_dump()
        except ValueError as exc:
            parser_error = str(exc)

    request_failed = response_error is not None
    runtime_parser_accepted = parsed_action is not None and not request_failed
    return {
        "raw_json_valid": raw_json_valid,
        "strict_action_envelope_valid": strict_envelope_valid,
        "normalized_parser_accepted": parsed_action is not None,
        "runtime_parser_accepted": runtime_parser_accepted,
        "parser_error": parser_error,
        "parser_normalized": parser_normalized,
        "predicted_action": parsed_action if runtime_parser_accepted else None,
        "predicted_action_from_raw": parsed_action,
        "predicted_kind": (
            parsed_action.get("kind") if runtime_parser_accepted and parsed_action else None
        ),
        "invalid_json": parser_error == "invalid_json",
        "invalid_envelope": parser_error == "invalid_action",
        "invalid_arguments": parser_error == "invalid_action_arguments",
        "empty_output": (raw is not None and not raw.strip())
        or (response_error or {}).get("code") == "empty_output",
        "truncated": finish_reason == "length",
        "request_failed": request_failed,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _args_match(record: dict[str, Any]) -> bool:
    predicted = record.get("predicted_action")
    if not predicted or predicted.get("kind") not in TOOL_KINDS:
        return False
    return canonical_json(predicted.get("arguments")) == canonical_json(
        record["gold_action"].get("arguments")
    )


def metrics_for(records: list[dict[str, Any]]) -> dict[str, Any]:
    target_count = len(records)
    accepted = [record for record in records if record.get("runtime_parser_accepted")]
    tool_records = [record for record in records if record["gold_kind"] in TOOL_KINDS]
    kind_matches = [
        record for record in accepted if record.get("predicted_kind") == record.get("gold_kind")
    ]
    tool_kind_matches = [
        record
        for record in tool_records
        if record.get("runtime_parser_accepted")
        and record.get("predicted_kind") == record.get("gold_kind")
    ]
    full_matches = [
        record for record in accepted if record.get("predicted_action") == record.get("gold_action")
    ]

    def validity(field: str) -> dict[str, Any]:
        count = sum(bool(record.get(field)) for record in records)
        return {"count": count, "denominator": target_count, "rate": _rate(count, target_count)}

    return {
        "target_count": target_count,
        "request_count": target_count,
        "request_failure_count": sum(bool(record.get("request_failed")) for record in records),
        "raw_json_validity": validity("raw_json_valid"),
        "strict_action_envelope_validity": validity("strict_action_envelope_valid"),
        "runtime_parser_acceptance": validity("runtime_parser_accepted"),
        "normalized_parser_acceptance": validity("normalized_parser_accepted"),
        "invalid_json_count": sum(bool(record.get("invalid_json")) for record in records),
        "raw_invalid_json_count": sum(
            not bool(record.get("raw_json_valid"))
            for record in records
            if record.get("raw_output") is not None and str(record.get("raw_output")).strip()
        ),
        "invalid_envelope_count": sum(bool(record.get("invalid_envelope")) for record in records),
        "invalid_arguments_count": sum(bool(record.get("invalid_arguments")) for record in records),
        "empty_output_count": sum(bool(record.get("empty_output")) for record in records),
        "truncated_count": sum(bool(record.get("truncated")) for record in records),
        "parser_normalization_count": sum(
            bool(record.get("parser_normalized")) for record in records
        ),
        "action_kind_exact_match": {
            "count": len(kind_matches),
            "denominator": target_count,
            "accuracy": _rate(len(kind_matches), target_count),
        },
        "tool_target_count": len(tool_records),
        "tool_argument_exact_match": {
            "count": sum(_args_match(record) for record in tool_records),
            "denominator": len(tool_records),
            "accuracy": _rate(
                sum(_args_match(record) for record in tool_records), len(tool_records)
            ),
            "definition": (
                "accepted predicted tool Action with canonical arguments equal; kind may differ"
            ),
        },
        "tool_argument_exact_match_on_kind_match": {
            "count": sum(_args_match(record) for record in tool_kind_matches),
            "denominator": len(tool_kind_matches),
            "accuracy": _rate(
                sum(_args_match(record) for record in tool_kind_matches), len(tool_kind_matches)
            ),
            "definition": (
                "canonical argument exact match conditioned on accepted predicted kind "
                "matching gold kind"
            ),
        },
        "full_action_exact_match": {
            "count": len(full_matches),
            "denominator": target_count,
            "accuracy": _rate(len(full_matches), target_count),
        },
        "tokens": {
            "input": sum(int(record.get("input_tokens") or 0) for record in records),
            "output": sum(int(record.get("output_tokens") or 0) for record in records),
            "mean_input": _rate(
                sum(int(record.get("input_tokens") or 0) for record in records), target_count
            ),
            "mean_output": _rate(
                sum(int(record.get("output_tokens") or 0) for record in records), target_count
            ),
        },
    }


def stratified_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = {}
    for source in sorted({record["teacher_source"] for record in records}):
        by_source[source] = metrics_for(
            [record for record in records if record["teacher_source"] == source]
        )
    by_template = {}
    for template in sorted({record["template_id"] for record in records}):
        by_template[template] = metrics_for(
            [record for record in records if record["template_id"] == template]
        )
    by_kind = {
        kind: metrics_for([record for record in records if record["gold_kind"] == kind])
        for kind in ACTION_KINDS
    }
    return {
        "teacher_source": by_source,
        "template": by_template,
        "action_kind": by_kind,
    }


def extract_response(
    response: dict[str, Any],
) -> tuple[Any, dict[str, Any] | None, str | None, dict[str, Any]]:
    error = response.get("error")
    response_error = error if isinstance(error, dict) else None
    choices = response.get("choices")
    raw_output: Any = response.get("raw_output")
    finish_reason = None
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        if isinstance(message.get("content"), str):
            raw_output = message["content"]
        finish_reason = choice.get("finish_reason")
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    telemetry = response.get("telemetry") if isinstance(response.get("telemetry"), dict) else {}
    return raw_output, response_error, finish_reason, usage | {"telemetry": telemetry}


def target_record(
    target: Target,
    *,
    stage: StageSpec,
    request_index: int,
    response: dict[str, Any],
    exception: BaseException | None = None,
) -> dict[str, Any]:
    if exception is not None:
        response = {
            "error": {
                "code": "generation_exception",
                "type": type(exception).__name__,
                "message": str(exception)[:500],
            }
        }
    raw_output, response_error, finish_reason, usage_data = extract_response(response)
    usage = {key: value for key, value in usage_data.items() if key != "telemetry"}
    telemetry = usage_data.get("telemetry") or {}
    classification = classify_output(
        raw_output=raw_output,
        response_error=response_error,
        finish_reason=finish_reason,
    )
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return {
        "request_index": request_index,
        "target_id": target.target_id,
        "dataset_row_index": target.dataset_row_index,
        "dataset_line_number": target.dataset_line_number,
        "example_id": target.example_id,
        "task_id": target.task_id,
        "assistant_message_index": target.assistant_message_index,
        "teacher_source": target.teacher_source,
        "template_id": target.template_id,
        "gold_history_message_count": len(target.history),
        "history_sha256": target.history_sha256,
        "current_gold_target_excluded_from_input": True,
        "gold_raw_action": target.raw_gold_action,
        "gold_action": target.gold_action,
        "gold_kind": target.gold_kind,
        "stage": stage.name,
        "checkpoint_stage": stage.checkpoint_stage,
        "global_step": stage.global_step,
        "raw_output": raw_output,
        "response_error": response_error,
        "finish_reason": finish_reason,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "usage": usage,
        "telemetry": telemetry,
        **classification,
    }


def model_load_tokenizer_hash(engine: Any) -> str:
    return sha256_bytes(canonical_json(engine.tokenizer.chat_template).encode("utf-8"))


def release_engine(engine: Any, torch: Any) -> None:
    model = getattr(engine, "model", None)
    if model is not None:
        try:
            del model
        except Exception:
            pass
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def stage_summary(
    *,
    stage: StageSpec,
    records: list[dict[str, Any]],
    config: Any,
    adapter_path: Path | None,
    adapter_sha256: str | None,
    model_load: dict[str, Any],
    resources: dict[str, Any],
    generation: dict[str, Any],
    parser: dict[str, Any],
) -> dict[str, Any]:
    metrics = metrics_for(records)
    complete = len(records) == 108 and metrics["request_failure_count"] == 0
    return {
        "schema_version": "1.0",
        "status": "complete" if complete else "failed",
        "stage": stage.name,
        "checkpoint_stage": stage.checkpoint_stage,
        "global_step": stage.global_step,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_weight_sha256": adapter_sha256,
        "model": {
            "model_id": config.model_id,
            "revision": config.revision,
            "dtype": config.dtype,
            "attention_backend": config.attention_backend,
        },
        "generation": generation,
        "parser": parser,
        "metrics": metrics,
        "stratified": stratified_metrics(records),
        "resources": resources,
        "model_load": model_load,
        "prediction_count": len(records),
        "prediction_file_is_first_generation_only": True,
        "format_retries_used": 0,
    }


def _metric_value(summary: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = summary
    for key in path:
        value = value[key]
    return value


def comparison_report(
    run_id: str,
    summaries: list[dict[str, Any]],
    *,
    generation: dict[str, Any],
    parser: dict[str, Any],
) -> dict[str, Any]:
    key_paths = {
        "raw_json_validity": ("metrics", "raw_json_validity", "rate"),
        "strict_action_envelope_validity": (
            "metrics",
            "strict_action_envelope_validity",
            "rate",
        ),
        "runtime_parser_acceptance": ("metrics", "runtime_parser_acceptance", "rate"),
        "action_kind_accuracy": ("metrics", "action_kind_exact_match", "accuracy"),
        "tool_argument_accuracy": (
            "metrics",
            "tool_argument_exact_match",
            "accuracy",
        ),
        "tool_argument_accuracy_on_kind_match": (
            "metrics",
            "tool_argument_exact_match_on_kind_match",
            "accuracy",
        ),
        "full_action_accuracy": ("metrics", "full_action_exact_match", "accuracy"),
    }
    stages: list[dict[str, Any]] = []
    for summary in summaries:
        values = {key: _metric_value(summary, path) for key, path in key_paths.items()}
        stages.append(
            {
                "stage": summary["stage"],
                "checkpoint_stage": summary["checkpoint_stage"],
                "global_step": summary["global_step"],
                "status": summary["status"],
                "metrics": values,
                "request_failure_count": summary["metrics"]["request_failure_count"],
            }
        )

    observations: list[str] = []
    if len(stages) >= 2:
        base = stages[0]["metrics"]
        epoch1 = stages[1]["metrics"]
        for key in key_paths:
            if base[key] is not None and epoch1[key] is not None:
                observations.append(f"base_to_epoch_1:{key}:{epoch1[key] - base[key]:+.6f}")
    if len(stages) == 4:
        for key in key_paths:
            values = [stage["metrics"][key] for stage in stages]
            if all(value is not None for value in values):
                if values[0] > values[1] > values[2] > values[3]:
                    observations.append(f"strict_stagewise_decline:{key}")
                elif values[0] < values[1] < values[2] < values[3]:
                    observations.append(f"strict_stagewise_increase:{key}")

    complete = len(summaries) == 4 and all(item["status"] == "complete" for item in summaries)
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": (
            "SFT_BEHAVIOR_ATTRIBUTION_COMPLETE" if complete else "SFT_BEHAVIOR_ATTRIBUTION_FAILED"
        ),
        "stages": stages,
        "generation": generation,
        "parser": parser,
        "observations": observations,
        "interpretation_guardrail": (
            "Observed teacher-forced differences describe local Action imitation only; "
            "they do not establish a causal learning-rate or epoch-count explanation and "
            "do not replace closed-loop validation."
        ),
    }


def _check_gpu_environment(run_dir: Path, preflight: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    from enterprise_agent_rl.gpu_guard import require_compute_node

    require_compute_node()
    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires_one_visible_cuda_device")
    name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    if "V100" not in name.upper() or capability != [7, 0]:
        raise RuntimeError(f"requires_v100_compatible_gpu:{name}:{capability}")
    properties = torch.cuda.get_device_properties(0)
    environment = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "gpu_name": name,
        "compute_capability": capability,
        "device_count": torch.cuda.device_count(),
        "total_memory_bytes": int(properties.total_memory),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_arch_list": torch.cuda.get_arch_list(),
        "nvidia_smi": command_result(["nvidia-smi"], timeout=30),
        "scontrol_job": command_result(
            ["scontrol", "show", "job", os.environ.get("SLURM_JOB_ID", "")], timeout=30
        ),
        "weights_loaded_on_cpu": False,
        "single_visible_gpu": True,
        "v100_gate_passed": True,
    }
    write_json(run_dir / "gpu-environment.json", environment)
    return torch, environment


def _recheck_before_gpu(
    *,
    root: Path,
    preflight: dict[str, Any],
    dataset: Path,
    metadata_path: Path,
    cache: Path,
    model_config: Path,
) -> tuple[Any, list[Target], dict[str, Any]]:
    from enterprise_agent_rl.model_server import ModelSettings

    if preflight.get("status") != "READY_FOR_GPU_SUBMISSION":
        raise RuntimeError("cpu_preflight_not_ready")
    config = ModelSettings.load(model_config)
    if config.model_id != MODEL_ID or config.revision != MODEL_REVISION:
        raise RuntimeError("gpu_model_identity_mismatch")
    _rows, _metadata, targets, data_checks = validate_rows_and_targets(dataset, metadata_path)
    if data_checks["target_fingerprint"] != preflight["data_checks"]["target_fingerprint"]:
        raise RuntimeError("target_fingerprint_changed_after_preflight")
    if sha256_file(cache / "qwen-cache-manifest.json") != preflight["cache"]["manifest_sha256"]:
        raise RuntimeError("cache_manifest_changed_after_preflight")
    for stage in STAGES:
        if stage.adapter_relative is None:
            continue
        weight = root / stage.adapter_relative / "adapter_model.safetensors"
        expected = preflight["checkpoints"][stage.checkpoint_stage]["adapter_weight_sha256"]
        if sha256_file(weight) != expected:
            raise RuntimeError(f"adapter_hash_changed_after_preflight:{stage.name}")
    return config, targets, data_checks


def run_gpu(
    *,
    root: Path,
    run_dir: Path,
    dataset: Path,
    metadata_path: Path,
    cache: Path,
    model_config: Path,
    timeout_s: float,
) -> int:
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("preflight_json_missing")
    preflight = json.loads(preflight_path.read_text())
    config, targets, data_checks = _recheck_before_gpu(
        root=root,
        preflight=preflight,
        dataset=dataset,
        metadata_path=metadata_path,
        cache=cache,
        model_config=model_config,
    )
    generation = fixed_generation_evidence(config)
    parser = parser_evidence()
    torch, gpu_environment = _check_gpu_environment(run_dir, preflight)
    environment_path = run_dir / "environment.json"
    environment = json.loads(environment_path.read_text()) if environment_path.is_file() else {}
    environment["phase"] = "gpu_running"
    environment["gpu_execution"] = gpu_environment
    environment["input_fingerprints_at_gpu_start"] = {
        "dataset_sha256": sha256_file(dataset),
        "metadata_sha256": sha256_file(metadata_path),
        "cache_manifest_sha256": sha256_file(cache / "qwen-cache-manifest.json"),
        "target_fingerprint": data_checks["target_fingerprint"],
    }
    write_json(environment_path, environment)

    # Delayed import prevents any CPU preflight path from importing a model
    # loader.  QwenEngine itself owns the pinned Transformers loading contract.
    from enterprise_agent_rl.model_server import CompletionRequest, Message, QwenEngine

    all_summaries: list[dict[str, Any]] = []
    run_error: BaseException | None = None
    try:
        for stage in STAGES:
            stage_dir = run_dir / stage.name
            stage_dir.mkdir(parents=True, exist_ok=False)
            predictions_path = stage_dir / "predictions.jsonl"
            adapter_path = root / stage.adapter_relative if stage.adapter_relative else None
            adapter_sha = (
                sha256_file(adapter_path / "adapter_model.safetensors") if adapter_path else None
            )
            engine = None
            stage_load: dict[str, Any] = {}
            records: list[dict[str, Any]] = []
            stage_started = time.monotonic()
            try:
                engine = QwenEngine(
                    config,
                    cache,
                    stage_dir / "engine",
                    adapter=adapter_path,
                )
                stage_load = engine.metadata
                if (
                    model_load_tokenizer_hash(engine)
                    != preflight["tokenizer"]["chat_template_sha256"]
                ):
                    raise RuntimeError(f"chat_template_hash_changed:{stage.name}")
                with predictions_path.open("w") as prediction_stream:
                    for request_index, target in enumerate(targets):
                        request = CompletionRequest(
                            model=config.model_id,
                            messages=[
                                Message.model_validate(message) for message in target.history
                            ],
                            temperature=0.0,
                            top_p=1.0,
                            max_tokens=512,
                            seed=17,
                            timeout_s=timeout_s,
                            stream=False,
                        )
                        try:
                            response = engine.generate(request)
                            record = target_record(
                                target,
                                stage=stage,
                                request_index=request_index,
                                response=response,
                            )
                        except BaseException as exc:  # preserve a per-target failure record
                            record = target_record(
                                target,
                                stage=stage,
                                request_index=request_index,
                                response={},
                                exception=exc,
                            )
                        write_jsonl_record(prediction_stream, record)
                        records.append(record)
                peak_allocated = max(
                    [
                        int(record.get("telemetry", {}).get("peak_allocated_bytes") or 0)
                        for record in records
                    ]
                    + [int(stage_load.get("load_peak_allocated_bytes") or 0)]
                )
                peak_reserved = max(
                    [
                        int(record.get("telemetry", {}).get("peak_reserved_bytes") or 0)
                        for record in records
                    ]
                    + [int(stage_load.get("load_peak_reserved_bytes") or 0)]
                )
                resources = {
                    **gpu_environment,
                    "stage_elapsed_s": time.monotonic() - stage_started,
                    "peak_memory_allocated_bytes": peak_allocated,
                    "peak_memory_reserved_bytes": peak_reserved,
                }
                summary = stage_summary(
                    stage=stage,
                    records=records,
                    config=config,
                    adapter_path=adapter_path,
                    adapter_sha256=adapter_sha,
                    model_load=stage_load,
                    resources=resources,
                    generation=generation,
                    parser=parser,
                )
                write_json(stage_dir / "summary.json", summary)
                all_summaries.append(summary)
            finally:
                if engine is not None:
                    release_engine(engine, torch)
                    engine = None
    except BaseException as exc:
        run_error = exc

    comparison = comparison_report(
        run_dir.name,
        all_summaries,
        generation=generation,
        parser=parser,
    )
    comparison["gpu_environment"] = gpu_environment
    comparison["completed_stage_count"] = len(all_summaries)
    comparison["fatal_error"] = (
        {"type": type(run_error).__name__, "message": str(run_error)[:1000]} if run_error else None
    )
    write_json(run_dir / "stage-comparison.json", comparison)
    environment["phase"] = "gpu_complete" if run_error is None else "gpu_failed"
    environment["stage_count_completed"] = len(all_summaries)
    environment["fatal_error"] = comparison["fatal_error"]
    write_json(environment_path, environment)
    return 0 if comparison["status"] == "SFT_BEHAVIOR_ATTRIBUTION_COMPLETE" else 2


def cpu_smoke() -> dict[str, Any]:
    """Run parser/schema/aggregation smoke checks without model weights."""

    gold = '{"kind":"schema","arguments":{}}'
    fenced = "```json\n" + gold + "\n```"
    accepted = classify_output(raw_output=fenced, response_error=None, finish_reason="stop")
    invalid = classify_output(
        raw_output='{"kind":"sql","arguments":{"query":42}}',
        response_error=None,
        finish_reason="stop",
    )
    if accepted["raw_json_valid"] or not accepted["runtime_parser_accepted"]:
        raise AssertionError("fence/raw-parser-smoke_failed")
    if not accepted["parser_normalized"] or accepted["predicted_kind"] != "schema":
        raise AssertionError("parser-normalization-smoke_failed")
    if not invalid["invalid_arguments"] or invalid["runtime_parser_accepted"] is True:
        raise AssertionError("invalid-arguments-smoke_failed")
    record = {
        "gold_action": {"kind": "schema", "arguments": {}},
        "gold_kind": "schema",
        "teacher_source": "deterministic_fixture_expert",
        "template_id": "schema",
        "raw_output": fenced,
        **accepted,
        "input_tokens": 12,
        "output_tokens": 8,
    }
    metrics = metrics_for([record])
    if metrics["runtime_parser_acceptance"]["count"] != 1:
        raise AssertionError("metrics-smoke_failed")
    return {
        "status": "passed",
        "model_weights_loaded": False,
        "gpu_used": False,
        "checks": [
            "raw_json_and_runtime_parser_are_distinct",
            "json_fence_normalization_uses_runtime_parser",
            "invalid_arguments_are_rejected",
            "metric_aggregation_is_schema_stable",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "cpu-smoke", "gpu"), required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/sft-data-train-teacher-v2/dataset.jsonl")
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/sft-data-train-teacher-v2/dataset-metadata.json"),
    )
    parser.add_argument("--cache", type=Path, default=Path("artifacts/model-cache"))
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen2.5-1.5b.toml")
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    if args.mode == "cpu-smoke":
        result = cpu_smoke()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.run_dir is None:
        parser.error("--run-dir is required for preflight and gpu")
    root = args.root.resolve()
    run_dir = args.run_dir.resolve()
    dataset = args.dataset.resolve()
    metadata = args.metadata.resolve()
    cache = args.cache.resolve()
    model_config = args.model_config.resolve()

    if args.mode == "preflight":
        if (run_dir / "preflight.json").exists():
            raise SystemExit(f"refusing_to_overwrite_existing_preflight:{run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            report, environment = build_preflight_report(
                root=root,
                dataset=dataset,
                metadata_path=metadata,
                cache=cache,
                model_config=model_config,
                run_dir=run_dir,
            )
        except BaseException as exc:
            report = {
                "schema_version": "1.0",
                "status": "SFT_BEHAVIOR_ATTRIBUTION_BLOCKED",
                "run_id": run_dir.name,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "cpu_preflight": {
                    "model_weights_loaded": False,
                    "gpu_used": False,
                    "python": sys.version,
                    "platform": platform.platform(),
                    "hostname": platform.node(),
                },
                "git": git_state(),
            }
            environment = {
                "schema_version": "1.0",
                "run_id": run_dir.name,
                "phase": "cpu_preflight_failed",
                "error": report["error"],
                "git_at_preflight": report["git"],
            }
            write_json(run_dir / "preflight.json", report)
            write_json(run_dir / "environment.json", environment)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 2
        write_json(run_dir / "preflight.json", report)
        write_json(run_dir / "environment.json", environment)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    try:
        return run_gpu(
            root=root,
            run_dir=run_dir,
            dataset=dataset,
            metadata_path=metadata,
            cache=cache,
            model_config=model_config,
            timeout_s=args.timeout_s,
        )
    except BaseException as exc:
        failure = {
            "schema_version": "1.0",
            "status": "SFT_BEHAVIOR_ATTRIBUTION_FAILED",
            "run_id": run_dir.name,
            "fatal_error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
        }
        write_json(run_dir / "stage-comparison.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
