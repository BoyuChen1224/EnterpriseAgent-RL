"""Explicit T-034 Qwen3 fixed-batch update contract and input audit.

The older :mod:`enterprise_agent_rl.grpo.config` contract intentionally stays
bound to the historical Qwen2.5/1.5B online entrypoint.  This module is the
small, fail-closed contract for the fixed Qwen3 batch used by T-034.  It does
not import torch and never opens validation/test payloads.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import tomllib
from pathlib import Path
from typing import Any

from .config import GrpoConfig, RunContext
from .io import sha256_file
from .qwen3_rollout import (
    QWEN3_ATTENTION_BACKEND,
    QWEN3_CONTEXT_TOKENS,
    QWEN3_MODEL_ID,
    QWEN3_MODULE_COUNT,
    QWEN3_PARENT_CONFIG_SHA256,
    QWEN3_PARENT_WEIGHT_SHA256,
    QWEN3_REVISION,
    QWEN3_TARGET_MODULES,
)

T034_TASK = "T-034/E-031"
SOURCE_RUN_RELATIVE = "artifacts/qwen3-grpo-rollout-signal/t033-v1"
SOURCE_ARTIFACT_MANIFEST_SHA256 = "e0c533b2c9af0bca5fa276001cd8fdc3bd4b41af200d977901db67edc59fc9ad"
SOURCE_TOKEN_ROLLOUTS_SHA256 = "ef63742a1150b9af4dd6a1fd9414ec74bdd5568e49614d568cb23b68ce59b3ad"
SOURCE_GROUP_METRICS_SHA256 = "a8e5cc5aca30a891c7965a2dba47b51a0b1092cec628e7e1b9b3cfa8ea8b396a"
SOURCE_CASE_RESULTS_SHA256 = "2ca0b9cc2a06e36e99fbaad46f30b6c2b3e9471441fd71b08dc4208ae41ee97b"
EXPECTED_TOKEN_ROWS = 118
EXPECTED_ACTIVE_RECORDS = 70
EXPECTED_ZERO_RECORDS = 48
EXPECTED_TOTAL_TOKENS = 4745
EXPECTED_ACTIVE_TOKENS = 3458
EXPECTED_MAX_SEQUENCE = 3519
EXPECTED_GROUP_COUNT = 4
EXPECTED_GROUP_SIZE = 4
EXPECTED_MODULE_COUNT = QWEN3_MODULE_COUNT


def _section(values: dict[str, Any], name: str) -> dict[str, Any]:
    result = values.get(name, {})
    if not isinstance(result, dict):
        raise ValueError(f"qwen3_update_section_not_table:{name}")
    return dict(result)


def _require(values: dict[str, Any], key: str) -> Any:
    if key not in values:
        raise ValueError(f"qwen3_update_key_missing:{key}")
    return values[key]


def _require_equal(values: dict[str, Any], key: str, expected: Any) -> None:
    if _require(values, key) != expected:
        raise ValueError(f"qwen3_update_value_mismatch:{key}")


def validate_qwen3_update_values(values: dict[str, Any]) -> None:
    """Validate the complete fixed T-034 identity without loading torch."""

    model = _section(values, "model")
    lora = _section(values, "lora")
    rollout = _section(values, "rollout")
    runtime = _section(values, "runtime")
    optimizer = _section(values, "optimizer")
    training = _section(values, "training")
    budget = _section(values, "budget")
    data = _section(values, "data")
    evaluation = _section(values, "evaluation")
    slurm = _section(values, "slurm")
    limits = _section(values, "limits")

    _require_equal(model, "model_id", QWEN3_MODEL_ID)
    _require_equal(model, "revision", QWEN3_REVISION)
    _require_equal(model, "dtype", "float16")
    _require_equal(model, "attention_backend", QWEN3_ATTENTION_BACKEND)
    _require_equal(model, "max_context_tokens", QWEN3_CONTEXT_TOKENS)
    _require_equal(model, "enable_thinking", False)
    _require_equal(model, "parent_adapter_sha256", QWEN3_PARENT_WEIGHT_SHA256)
    _require_equal(model, "parent_adapter_config_sha256", QWEN3_PARENT_CONFIG_SHA256)
    for key in ("cache_dir", "parent_adapter"):
        if not isinstance(_require(model, key), str) or not _require(model, key):
            raise ValueError(f"qwen3_update_path_missing:{key}")

    _require_equal(lora, "rank", 8)
    _require_equal(lora, "alpha", 16.0)
    _require_equal(lora, "dropout", 0.0)
    _require_equal(lora, "module_count", EXPECTED_MODULE_COUNT)
    _require_equal(lora, "target_modules", list(QWEN3_TARGET_MODULES))

    _require_equal(rollout, "rounds", 1)
    _require_equal(rollout, "updates", 2)
    _require_equal(rollout, "samples_per_group", EXPECTED_GROUP_SIZE)
    _require_equal(rollout, "group_count", EXPECTED_GROUP_COUNT)
    _require_equal(rollout, "generation_enabled", False)

    expected_runtime = {
        "seed": 17,
        "max_actions": 20,
        "max_tool_calls": 14,
        "episode_timeout_s": 600,
        "inference_timeout_s": 120,
        "sql_timeout_s": 2,
        "sql_max_rows": 100,
        "sql_max_bytes": 65536,
        "sql_memory_mb": 256,
        "step_penalty": 0.0,
        "repeat_penalty": 0.0,
        "error_penalty": 0.0,
        "recovery_bonus": 0.0,
    }
    for key, expected in expected_runtime.items():
        _require_equal(runtime, key, expected)

    expected_optimizer = {
        "learning_rate": 2e-6,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "clip_norm": 1.0,
        "epsilon": 0.2,
        "beta": 0.0,
        "grad_scaler_init": 1024.0,
        "grad_scaler_backoff": 0.5,
        "grad_scaler_growth_interval": 1000,
        "max_amp_skip_replays": 2,
        "checkpoint_use_reentrant": True,
        "enable_input_require_grads": True,
    }
    for key, expected in expected_optimizer.items():
        _require_equal(optimizer, key, expected)

    expected_training = {
        "seed": 42,
        "microbatch": 1,
        "successful_logical_updates": 2,
        "total_token_denominator": EXPECTED_TOTAL_TOKENS,
        "input_max_sequence_tokens": EXPECTED_MAX_SEQUENCE,
    }
    for key, expected in expected_training.items():
        _require_equal(training, key, expected)

    expected_budget = {
        "run_hard_seconds": 10800,
        "job_hard_seconds": 3600,
        "soft_seconds": 3480,
        "cleanup_reserve_seconds": 120,
        "gpu_count": 1,
    }
    for key, expected in expected_budget.items():
        _require_equal(budget, key, expected)

    for key in (
        "source_run",
        "token_rollouts",
        "group_metrics",
        "case_results",
        "artifact_manifest",
    ):
        if not isinstance(_require(data, key), str) or not _require(data, key):
            raise ValueError(f"qwen3_update_data_path_missing:{key}")
    _require_equal(data, "artifact_manifest_sha256", SOURCE_ARTIFACT_MANIFEST_SHA256)
    _require_equal(data, "token_rollouts_sha256", SOURCE_TOKEN_ROLLOUTS_SHA256)
    _require_equal(data, "group_metrics_sha256", SOURCE_GROUP_METRICS_SHA256)
    _require_equal(data, "case_results_sha256", SOURCE_CASE_RESULTS_SHA256)

    if _require(evaluation, "validation_manifest_sha256") != (
        "056777038235f49cd1f553879ddfd5ebe8379f77e18a8da46804f9dc28e47c2a"
    ):
        raise ValueError("qwen3_update_validation_manifest_identity_mismatch")
    for key in ("validation_source_run", "validation_public_panel", "validation_private"):
        if not isinstance(_require(evaluation, key), str) or not _require(evaluation, key):
            raise ValueError(f"qwen3_update_evaluation_path_missing:{key}")
    generation = _section(evaluation, "generation")
    for key, expected in {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 512,
        "format_retries": 1,
        "parse_mode": "json_fence",
        "seed": 17,
        "enable_thinking": False,
    }.items():
        _require_equal(generation, key, expected)

    if _require(slurm, "account") != "pi_daipeishan" or _require(slurm, "qos") != "gpuq":
        raise ValueError("qwen3_update_slurm_identity_mismatch")
    if tuple(_require(slurm, "partitions")) != ("gpu2Q", "gpu4Q", "gpu4RQ", "gpu8Q"):
        raise ValueError("qwen3_update_partitions_mismatch")
    if int(_require(slurm, "gpus")) != 1 or int(_require(slurm, "cpus_per_task")) != 4:
        raise ValueError("qwen3_update_slurm_resource_mismatch")
    if int(_require(slurm, "memory_gb")) != 64 or int(_require(slurm, "time_minutes")) != 60:
        raise ValueError("qwen3_update_slurm_time_or_memory_mismatch")

    expected_limits = {
        "forward_backward": 420,
        "optimizer_update_attempts": 6,
        "successful_optimizer_updates": 4,
        "amp_skip_replays": 2,
        "reload_forward": 8,
        "model_requests": 1600,
        "tool_actions": 600,
        "runtime_workers": 600,
        "verifier_calls": 48,
        "verifier_sql": 96,
    }
    for key, expected in expected_limits.items():
        _require_equal(limits, key, expected)


def load_qwen3_update_config(
    config_path: Path,
    context: RunContext,
    *,
    cache_dir: Path | None = None,
    parent_adapter: Path | None = None,
    final_delta: Path | None = None,
) -> tuple[GrpoConfig, RunContext]:
    """Load the explicit T-034 config and resolve only approved model paths."""

    config_path = config_path.expanduser().resolve()
    values = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("qwen3_update_config_root_not_table")
    validate_qwen3_update_values(values)
    resolved_context = context.resolved()
    model = _section(values, "model")
    model["cache_dir"] = str(
        (cache_dir or _resolve(model["cache_dir"], resolved_context)).expanduser().resolve()
    )
    model["parent_adapter"] = str(
        (parent_adapter or _resolve(model["parent_adapter"], resolved_context))
        .expanduser()
        .resolve()
    )
    values["model"] = model
    evaluation = _section(values, "evaluation")
    if final_delta is not None:
        delta = final_delta.expanduser().resolve()
        weight = delta / "adapter_model.safetensors"
        config = delta / "adapter_config.json"
        if not weight.is_file() or not config.is_file():
            raise ValueError("qwen3_update_final_delta_files_missing")
        evaluation["rl_delta_adapter"] = str(delta)
        evaluation["rl_delta_sha256"] = sha256_file(weight)
        evaluation["rl_delta_config_sha256"] = sha256_file(config)
    values["evaluation"] = evaluation
    resolved_context = RunContext(resolved_context.repo_root, resolved_context.run_dir, config_path)
    return (
        GrpoConfig(
            raw=values,
            model=model,
            lora=_section(values, "lora"),
            rollout=_section(values, "rollout"),
            runtime=_section(values, "runtime"),
            optimizer=_section(values, "optimizer"),
            budget=_section(values, "budget"),
            data=_section(values, "data"),
            evaluation=evaluation,
            slurm=_section(values, "slurm"),
            limits=_section(values, "limits"),
        ),
        resolved_context,
    )


def _resolve(value: str, context: RunContext) -> Path:
    path = Path(value)
    return path if path.is_absolute() else context.repo_root / path


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _finite_sequence(values: Any, label: str) -> None:
    if (
        not isinstance(values, list)
        or not values
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in values
        )
    ):
        raise ValueError(f"qwen3_update_nonfinite_sequence:{label}")


def _verify_source_chain(source: Path, token_rows: list[dict[str, Any]]) -> dict[str, Any]:
    request_ids: set[str] = set()
    response_ids: set[str] = set()
    episode_ids: set[str] = set()
    for row in token_rows:
        episode_id = row.get("episode_id")
        request_id = row.get("request_id")
        response_id = row.get("response_id")
        metadata = row.get("metadata")
        if not all(
            isinstance(item, str) and item for item in (episode_id, request_id, response_id)
        ):
            raise ValueError("qwen3_update_source_identity_missing")
        if request_id in request_ids or response_id in response_ids:
            raise ValueError("qwen3_update_duplicate_physical_request")
        request_ids.add(request_id)
        response_ids.add(response_id)
        episode_ids.add(episode_id)
        if not isinstance(metadata, dict):
            raise ValueError("qwen3_update_source_metadata_missing")
        for path_key, hash_key in (("request_path", "request_sha256"),):
            relative = metadata.get(path_key)
            expected_hash = metadata.get(hash_key)
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise ValueError(f"qwen3_update_source_{path_key}_missing")
            path = source / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"qwen3_update_source_{path_key}_hash_mismatch:{request_id}")
        response_relative = metadata.get("response_path")
        if not isinstance(response_relative, str):
            raise ValueError("qwen3_update_source_response_path_missing")
        response_path = source / response_relative
        if not response_path.is_file():
            raise ValueError(f"qwen3_update_source_response_path_missing:{request_id}")
        event_relative = metadata.get("engine_event_path")
        if not isinstance(event_relative, str):
            raise ValueError("qwen3_update_source_engine_event_missing")
        event_path = source / event_relative
        if not event_path.is_file():
            raise ValueError(f"qwen3_update_source_engine_event_missing:{request_id}")
        event = _read_json(event_path)
        if (
            event.get("request_id") != request_id
            or event.get("response_id") != response_id
            or event.get("request_sha256") != metadata.get("request_sha256")
            or event.get("response_sha256") != sha256_file(response_path)
        ):
            raise ValueError(f"qwen3_update_source_event_binding_mismatch:{request_id}")
    journal = source / "engine-evidence" / "journal.jsonl"
    if not journal.is_file():
        raise ValueError("qwen3_update_source_journal_missing")
    journal_rows = _read_jsonl(journal)
    intents = {
        row.get("request_id")
        for row in journal_rows
        if row.get("event") == "attempt_intent_persisted"
    }
    completed = {
        row.get("request_id") for row in journal_rows if row.get("event") == "attempt_completed"
    }
    if intents != request_ids or completed != request_ids:
        raise ValueError("qwen3_update_source_journal_binding_mismatch")
    return {
        "token_rows": len(token_rows),
        "unique_episodes": len(episode_ids),
        "physical_requests": len(request_ids),
        "journal_intents": len(intents),
        "journal_completed": len(completed),
    }


def _group_advantages(result_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        if row.get("record_type") != "RESULT" or row.get("status") != "scored":
            raise ValueError(f"qwen3_update_result_not_scored:{row.get('episode_id')}")
        reward = row.get("reward_value")
        if (
            not isinstance(reward, int | float)
            or isinstance(reward, bool)
            or not math.isfinite(float(reward))
        ):
            raise ValueError(f"qwen3_update_reward_invalid:{row.get('episode_id')}")
        metadata_reward = (row.get("metadata") or {}).get("reward", {}).get("value")
        if metadata_reward != reward:
            raise ValueError(f"qwen3_update_reward_metadata_mismatch:{row.get('episode_id')}")
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("qwen3_update_result_group_missing")
        by_group.setdefault(group_id, []).append(row)
    if len(by_group) != EXPECTED_GROUP_COUNT:
        raise ValueError("qwen3_update_group_count_mismatch")
    groups: dict[str, dict[str, Any]] = {}
    for group_id, rows in by_group.items():
        rows.sort(key=lambda row: int(row["sample_index"]))
        if [int(row["sample_index"]) for row in rows] != list(range(EXPECTED_GROUP_SIZE)):
            raise ValueError(f"qwen3_update_sample_index_mismatch:{group_id}")
        rewards = [float(row["reward_value"]) for row in rows]
        mean = statistics.mean(rewards)
        sample_std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        advantages = [
            0.0 if sample_std == 0.0 else (reward - mean) / (sample_std + 1e-4)
            for reward in rewards
        ]
        groups[group_id] = {
            "group_id": group_id,
            "episode_ids": [row["episode_id"] for row in rows],
            "sample_indices": [int(row["sample_index"]) for row in rows],
            "reward_values": rewards,
            "mean": mean,
            "sample_std": sample_std,
            "advantages": advantages,
            "zero_advantage": sum(value == 0.0 for value in advantages),
        }
    return groups


def _verify_group_metrics(group_metrics: dict[str, Any], groups: dict[str, dict[str, Any]]) -> None:
    declared = group_metrics.get("groups")
    if not isinstance(declared, list) or len(declared) != EXPECTED_GROUP_COUNT:
        raise ValueError("qwen3_update_group_metrics_shape_mismatch")
    declared_by_id = {row.get("group_id"): row for row in declared}
    if set(declared_by_id) != set(groups):
        raise ValueError("qwen3_update_group_metrics_group_binding_mismatch")
    for group_id, observed in groups.items():
        row = declared_by_id[group_id]
        if (
            row.get("complete") is not True
            or int(row.get("expected_denominator", -1)) != EXPECTED_GROUP_SIZE
            or row.get("episode_ids") != observed["episode_ids"]
            or row.get("reward_values") != observed["reward_values"]
            or int(row.get("reward_count", -1)) != EXPECTED_GROUP_SIZE
        ):
            raise ValueError(f"qwen3_update_group_metrics_binding_mismatch:{group_id}")
        for key in ("sample_std",):
            if not math.isclose(
                float(row.get(key)), float(observed[key]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"qwen3_update_group_metrics_value_mismatch:{group_id}:{key}")
        declared_advantages = row.get("advantages")
        if (
            not isinstance(declared_advantages, list)
            or len(declared_advantages) != EXPECTED_GROUP_SIZE
        ):
            raise ValueError(f"qwen3_update_group_metrics_advantage_shape:{group_id}")
        if any(
            not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(declared_advantages, observed["advantages"], strict=True)
        ):
            raise ValueError(f"qwen3_update_group_metrics_advantage_mismatch:{group_id}")


def validate_train_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the post-assignment batch before any model forward/backward."""

    if len(rows) != EXPECTED_TOKEN_ROWS:
        raise ValueError(f"qwen3_update_batch_row_count:{len(rows)}")
    request_ids: set[str] = set()
    total_tokens = active_tokens = active_records = zero_records = 0
    max_sequence = 0
    for row in rows:
        if row.get("record_type") != "TOKEN":
            raise ValueError("qwen3_update_batch_record_type")
        advantage = row.get("advantage")
        if (
            not isinstance(advantage, int | float)
            or isinstance(advantage, bool)
            or not math.isfinite(float(advantage))
        ):
            raise ValueError(f"qwen3_update_batch_advantage_invalid:{row.get('request_id')}")
        generated = row.get("generated_token_ids")
        prompt = row.get("prompt_token_ids")
        mask = row.get("completion_mask")
        old_logps = row.get("old_per_token_logps")
        if (
            not isinstance(prompt, list)
            or not prompt
            or not isinstance(generated, list)
            or not generated
        ):
            raise ValueError(f"qwen3_update_batch_tokens_missing:{row.get('request_id')}")
        if (
            not isinstance(mask, list)
            or len(mask) != len(generated)
            or any(value not in (0, 1) for value in mask)
        ):
            raise ValueError(f"qwen3_update_batch_mask_invalid:{row.get('request_id')}")
        if not isinstance(old_logps, list) or len(old_logps) != len(generated):
            raise ValueError(f"qwen3_update_batch_logprob_length:{row.get('request_id')}")
        _finite_sequence(old_logps, f"old_logps:{row.get('request_id')}")
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or request_id in request_ids:
            raise ValueError("qwen3_update_batch_duplicate_request")
        request_ids.add(request_id)
        mask_tokens = sum(int(value) for value in mask)
        if mask_tokens <= 0:
            raise ValueError(f"qwen3_update_batch_empty_completion:{request_id}")
        total_tokens += mask_tokens
        if float(advantage) == 0.0:
            zero_records += 1
        else:
            active_records += 1
            active_tokens += mask_tokens
        max_sequence = max(max_sequence, len(prompt) + len(generated))
    observed = {
        "records": len(rows),
        "active_records": active_records,
        "zero_records": zero_records,
        "total_tokens": total_tokens,
        "active_tokens": active_tokens,
        "max_full_sequence_tokens": max_sequence,
    }
    expected = {
        "records": EXPECTED_TOKEN_ROWS,
        "active_records": EXPECTED_ACTIVE_RECORDS,
        "zero_records": EXPECTED_ZERO_RECORDS,
        "total_tokens": EXPECTED_TOTAL_TOKENS,
        "active_tokens": EXPECTED_ACTIVE_TOKENS,
        "max_full_sequence_tokens": EXPECTED_MAX_SEQUENCE,
    }
    if observed != expected:
        raise ValueError(f"qwen3_update_batch_counts_mismatch:{observed}")
    return observed


def build_frozen_train_batch(source_run: Path, output_path: Path) -> dict[str, Any]:
    """Copy the immutable token fields into a new run and assign episode rewards."""

    source_run = source_run.expanduser().resolve()
    source_paths = {
        "artifact_manifest": source_run / "artifact-manifest.json",
        "token_rollouts": source_run / "token-rollouts.jsonl",
        "group_metrics": source_run / "group-metrics.json",
        "case_results": source_run / "case-results.jsonl",
        "episodes": source_run / "raw" / "episodes.jsonl",
    }
    expected_hashes = {
        "artifact_manifest": SOURCE_ARTIFACT_MANIFEST_SHA256,
        "token_rollouts": SOURCE_TOKEN_ROLLOUTS_SHA256,
        "group_metrics": SOURCE_GROUP_METRICS_SHA256,
        "case_results": SOURCE_CASE_RESULTS_SHA256,
    }
    for key, expected in expected_hashes.items():
        path = source_paths[key]
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"qwen3_update_source_hash_mismatch:{key}")
    manifest = _read_json(source_paths["artifact_manifest"])
    manifest_files = {row.get("path"): row.get("sha256") for row in manifest.get("files", [])}
    for relative, expected in {
        "token-rollouts.jsonl": SOURCE_TOKEN_ROLLOUTS_SHA256,
        "group-metrics.json": SOURCE_GROUP_METRICS_SHA256,
        "case-results.jsonl": SOURCE_CASE_RESULTS_SHA256,
    }.items():
        if manifest_files.get(relative) != expected:
            raise ValueError(f"qwen3_update_artifact_manifest_binding:{relative}")
    token_rows = _read_jsonl(source_paths["token_rollouts"])
    if len(token_rows) != EXPECTED_TOKEN_ROWS or any(
        row.get("advantage") is not None for row in token_rows
    ):
        raise ValueError("qwen3_update_source_advantage_contract_failed")
    chain = _verify_source_chain(source_run, token_rows)
    episode_rows = _read_jsonl(source_paths["episodes"])
    episode_ids = {row.get("episode_id") for row in episode_rows}
    token_episode_ids = {row.get("episode_id") for row in token_rows}
    if episode_ids != token_episode_ids or len(episode_rows) != len(token_episode_ids):
        raise ValueError("qwen3_update_episode_source_binding_mismatch")
    result_rows = _read_jsonl(source_paths["case_results"])
    if len(result_rows) != EXPECTED_GROUP_COUNT * EXPECTED_GROUP_SIZE:
        raise ValueError("qwen3_update_result_row_count")
    result_by_episode = {row.get("episode_id"): row for row in result_rows}
    if len(result_by_episode) != len(result_rows):
        raise ValueError("qwen3_update_duplicate_result_episode")
    groups = _group_advantages(result_rows)
    _verify_group_metrics(_read_json(source_paths["group_metrics"]), groups)
    episode_to_group: dict[str, tuple[float, float]] = {}
    for group in groups.values():
        for episode_id, advantage in zip(group["episode_ids"], group["advantages"], strict=True):
            episode_to_group[episode_id] = (float(advantage), float(group["sample_std"]))
    output_rows: list[dict[str, Any]] = []
    for row in token_rows:
        episode_id = row["episode_id"]
        if episode_id not in result_by_episode or episode_id not in episode_to_group:
            raise ValueError(f"qwen3_update_episode_result_missing:{episode_id}")
        result = result_by_episode[episode_id]
        if row.get("group_id") != result.get("group_id") or int(row.get("sample_index")) != int(
            result.get("sample_index")
        ):
            raise ValueError(f"qwen3_update_episode_group_binding:{episode_id}")
        advantage, sample_std = episode_to_group[episode_id]
        copied = dict(row)
        copied["advantage"] = advantage
        copied["reward_value"] = float(result["reward_value"])
        copied["source_token_row_sha256"] = _sha_value(row)
        copied["source_run"] = str(source_run)
        copied["source_artifact_manifest_sha256"] = SOURCE_ARTIFACT_MANIFEST_SHA256
        copied["group_sample_std"] = sample_std
        copied["completion_mask_tokens"] = sum(int(value) for value in row["completion_mask"])
        copied["full_sequence_tokens"] = len(row["prompt_token_ids"]) + len(
            row["generated_token_ids"]
        )
        output_rows.append(copied)
    counts = validate_train_batch(output_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in output_rows)
        + "\n",
        encoding="utf-8",
    )
    group_summary = {
        group_id: {key: value for key, value in group.items() if key not in {"group_id"}}
        for group_id, group in groups.items()
    }
    return {
        "schema_version": "t034-train-batch-v1",
        "task": T034_TASK,
        "source_run": str(source_run),
        "source_hashes": expected_hashes,
        "source_chain": chain,
        "groups": group_summary,
        "counts": counts,
        "advantage_definition": (
            "episode RESULT reward; (reward-group_mean)/(sample_std+1e-4), "
            "sample_std=sample standard deviation; zero std -> 0"
        ),
        "old_advantage_fields_all_null": True,
        "input_fields_preserved": [
            "prompt_token_ids",
            "generated_token_ids",
            "old_per_token_logps",
            "completion_mask",
            "response_mask",
            "metadata",
            "request_id",
            "response_id",
        ],
        "zero_advantage_records_retained": True,
        "zero_advantage_gradient_policy": (
            "skip record forward/backward; retain in full token denominator"
        ),
        "output_sha256": sha256_file(output_path),
        "output_path": str(output_path),
    }


__all__ = [
    "EXPECTED_ACTIVE_RECORDS",
    "EXPECTED_ACTIVE_TOKENS",
    "EXPECTED_MAX_SEQUENCE",
    "EXPECTED_MODULE_COUNT",
    "EXPECTED_TOKEN_ROWS",
    "EXPECTED_TOTAL_TOKENS",
    "EXPECTED_ZERO_RECORDS",
    "SOURCE_ARTIFACT_MANIFEST_SHA256",
    "SOURCE_CASE_RESULTS_SHA256",
    "SOURCE_GROUP_METRICS_SHA256",
    "SOURCE_RUN_RELATIVE",
    "SOURCE_TOKEN_ROLLOUTS_SHA256",
    "T034_TASK",
    "build_frozen_train_batch",
    "load_qwen3_update_config",
    "validate_qwen3_update_values",
    "validate_train_batch",
]
