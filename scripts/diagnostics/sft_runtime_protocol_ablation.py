#!/usr/bin/env python3
"""Train-only closed-loop SFT Runtime protocol ablation.

This diagnostic intentionally lives outside the production Runtime.  It keeps
the public task, Observation, tool result, parser and verifier contracts
unchanged while making three protocol differences explicit:

* whether the original reset Observation is retained in later requests;
* whether accepted assistant Actions are serialized canonically;
* how many format retries are allowed.

The manifest reader is deliberately streaming and stops after the twenty
train records.  It does not deserialize or open validation/test task payloads.
Reference SQL is held only by the private verifier-side ``BenchmarkTask`` and
is never passed to the policy, Environment, request logger, or trajectory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from enterprise_agent_rl.agent import SYSTEM_PROMPT, run_episode
from enterprise_agent_rl.benchmark import canonical_hash, database_content_fingerprint
from enterprise_agent_rl.common import Action, Observation, PublicTask, Settings
from enterprise_agent_rl.data import BenchmarkTask
from enterprise_agent_rl.model_protocol import GenerationSettings, ModelFailure, parse_action
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.sft_data import _json as training_canonical_json
from enterprise_agent_rl.sft_training import load_dataset_rows, sha256_file
from enterprise_agent_rl.trajectories import StepRecord, Trajectory

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ADAPTER_RELATIVE = Path("artifacts/sft-pilot-20390015/adapter")
EXPECTED_ADAPTER_SHA256 = "a7d252f782a541dfc697055d43c7a25768d99633b6d5c3335131d00a6247bd77"
EXPECTED_TRAIN_COUNT = 20
EXPECTED_DATABASES = ("train-0", "train-1")
EXPECTED_TEMPLATES = (
    "category",
    "constraints",
    "customer",
    "empty",
    "inactive",
    "product",
    "recovery",
    "region",
    "schema",
    "top",
)
ACTION_KINDS = ("plan", "schema", "dictionary", "sql", "check", "finish")
PARSER_MODE = "json_fence"
GROUP_NAMES = (
    "baseline",
    "initial-observation",
    "canonical-history",
    "no-format-retry",
    "full-alignment",
)


def canonical_json(value: Any) -> str:
    """Use the exact JSON serializer used by the SFT data formatter."""

    return training_canonical_json(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def command_result(command: list[str], timeout: float = 30.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "error": type(exc).__name__,
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def git_state() -> dict[str, Any]:
    return {
        "commit": command_result(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "branch": command_result(["git", "branch", "--show-current"])["stdout"].strip(),
        "status": command_result(["git", "status", "--short", "--branch"])["stdout"],
    }


@dataclass(frozen=True)
class ProtocolGroup:
    name: str
    retain_initial_observation: bool
    canonical_history: bool
    format_retries: int

    def model_dump(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "retain_initial_observation": self.retain_initial_observation,
            "canonical_history": self.canonical_history,
            "format_retries": self.format_retries,
        }


GROUPS = (
    ProtocolGroup("baseline", False, False, 1),
    ProtocolGroup("initial-observation", True, False, 1),
    ProtocolGroup("canonical-history", False, True, 1),
    ProtocolGroup("no-format-retry", False, False, 0),
    ProtocolGroup("full-alignment", True, True, 0),
)


@dataclass(frozen=True)
class TrainCase:
    """Public task plus verifier-only private task data."""

    public_task: PublicTask
    verifier_task: BenchmarkTask
    database_id: str
    database: Path
    template_id: str
    teacher_provenance: str


@dataclass(frozen=True)
class BackendRequest:
    model: str
    messages: list[dict[str, str]]
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    timeout_s: float


class GenerationBackend(Protocol):
    def generate(self, request: BackendRequest) -> dict[str, Any]: ...


def _read_until_records_array(stream: Any) -> None:
    """Seek to the frozen manifest's top-level records array without parsing it."""

    marker = '"records"'
    window = ""
    while True:
        char = stream.read(1)
        if not char:
            raise ValueError("manifest_records_array_missing")
        window = (window + char)[-len(marker) :]
        if window == marker:
            break
    while True:
        char = stream.read(1)
        if not char:
            raise ValueError("manifest_records_colon_missing")
        if not char.isspace():
            if char != ":":
                raise ValueError("manifest_records_colon_missing")
            break
    while True:
        char = stream.read(1)
        if not char:
            raise ValueError("manifest_records_bracket_missing")
        if not char.isspace():
            if char != "[":
                raise ValueError("manifest_records_bracket_missing")
            return


def _read_one_json_value(stream: Any) -> Any:
    decoder = json.JSONDecoder()
    chars: list[str] = []
    while True:
        char = stream.read(1)
        if not char:
            raise ValueError("manifest_train_record_truncated")
        if not chars and char.isspace():
            continue
        chars.append(char)
        try:
            value, end = decoder.raw_decode("".join(chars))
        except json.JSONDecodeError:
            continue
        if end != len(chars):
            raise ValueError("manifest_record_decoder_leftover")
        return value


def stream_first_train_records(manifest_path: Path) -> Iterator[dict[str, Any]]:
    """Yield only the first frozen train records and stop before other splits.

    The project manifest is ordered train, validation, test.  This function
    proves the boundary by validating the split on every yielded object and by
    stopping immediately after the expected train count.  No non-train object
    is passed to ``json.loads``/``raw_decode``.
    """

    with manifest_path.open("r", encoding="utf-8") as stream:
        _read_until_records_array(stream)
        for index in range(EXPECTED_TRAIN_COUNT):
            record = _read_one_json_value(stream)
            if not isinstance(record, dict):
                raise ValueError("manifest_train_record_not_object")
            task_payload = record.get("task")
            if not isinstance(task_payload, dict) or task_payload.get("split") != "train":
                raise ValueError("manifest_train_record_not_train")
            public = task_payload.get("public")
            if not isinstance(public, dict) or not str(public.get("task_id", "")).startswith(
                "train-"
            ):
                raise ValueError("manifest_train_task_id_mismatch")
            yield record
            if index + 1 < EXPECTED_TRAIN_COUNT:
                separator = stream.read(1)
                while separator and separator.isspace():
                    separator = stream.read(1)
                if separator != ",":
                    raise ValueError("manifest_train_record_separator_missing")


def _dataset_public_task(row: dict[str, Any]) -> PublicTask:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("dataset_public_messages_missing")
    task_message = messages[1]
    if task_message.get("role") != "user" or not isinstance(task_message.get("content"), str):
        raise ValueError("dataset_public_task_message_invalid")
    return PublicTask(task_id=str(row["task_id"]), prompt=task_message["content"])


def load_train_only_cases(
    *,
    manifest_path: Path,
    dataset_path: Path,
    metadata_path: Path,
    database_root: Path,
) -> tuple[list[TrainCase], dict[str, Any]]:
    """Load public train tasks and private verifier data without forbidden payloads."""

    rows, metadata = load_dataset_rows(dataset_path, metadata_path)
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != metadata.get("manifest_sha256"):
        raise ValueError("frozen_manifest_sidecar_mismatch")

    public_by_id: dict[str, tuple[PublicTask, str]] = {}
    for row in rows:
        task_id = row.get("task_id")
        source = row.get("source")
        if not isinstance(task_id, str) or not isinstance(source, dict):
            raise ValueError("dataset_task_source_invalid")
        public_by_id[task_id] = (
            _dataset_public_task(row),
            str(source.get("trajectory_source", "unknown")),
        )
    if set(public_by_id) != set(metadata.get("source_manifest_task_ids", public_by_id)):
        # The v2 metadata does not carry this field; the explicit cardinality
        # and train prefix checks below are the fallback for that artifact.
        if len(public_by_id) != EXPECTED_TRAIN_COUNT:
            raise ValueError("dataset_train_task_set_invalid")
    if len(public_by_id) != EXPECTED_TRAIN_COUNT:
        raise ValueError("dataset_train_task_count_invalid")

    records: list[dict[str, Any]] = list(stream_first_train_records(manifest_path))
    if len(records) != EXPECTED_TRAIN_COUNT:
        raise ValueError("streamed_train_record_count_invalid")
    cases: list[TrainCase] = []
    database_hashes: dict[str, str] = {}
    for record in records:
        task_payload = record["task"]
        verifier_task = BenchmarkTask.model_validate(task_payload)
        task_id = verifier_task.public.task_id
        if task_id not in public_by_id:
            raise ValueError(f"manifest_dataset_task_mismatch:{task_id}")
        public_task, provenance = public_by_id[task_id]
        if verifier_task.public != public_task:
            raise ValueError(f"public_task_mismatch:{task_id}")
        if not task_id.startswith("train-"):
            raise ValueError("non_train_task_in_cases")
        database_id = record.get("database_id")
        if database_id not in EXPECTED_DATABASES:
            raise ValueError(f"non_train_database_in_cases:{database_id}")
        fields = {key: value for key, value in record.items() if key != "data_fingerprint"}
        if canonical_hash(fields) != record.get("data_fingerprint"):
            raise ValueError(f"train_record_fingerprint_mismatch:{task_id}")
        database = database_root / f"{database_id}.sqlite3"
        if database_id not in database_hashes:
            if not database.is_file():
                raise FileNotFoundError(database)
            database_hashes[database_id] = database_content_fingerprint(database)
        if record.get("database_sha256") != database_hashes[database_id]:
            raise ValueError(f"train_database_fingerprint_mismatch:{database_id}")
        cases.append(
            TrainCase(
                public_task=public_task,
                verifier_task=verifier_task,
                database_id=database_id,
                database=database,
                template_id=str(record.get("template")),
                teacher_provenance=provenance,
            )
        )

    cases.sort(key=lambda item: item.public_task.task_id)
    task_ids = [case.public_task.task_id for case in cases]
    templates = sorted({case.template_id for case in cases})
    databases = sorted({case.database_id for case in cases})
    if len(set(task_ids)) != EXPECTED_TRAIN_COUNT:
        raise ValueError("duplicate_train_task_ids")
    if templates != list(EXPECTED_TEMPLATES):
        raise ValueError(f"train_template_set_mismatch:{templates}")
    if databases != list(EXPECTED_DATABASES):
        raise ValueError(f"train_database_set_mismatch:{databases}")
    boundary = {
        "loader": "stream_first_train_records_only",
        "manifest_path": str(manifest_path),
        "manifest_sha256_from_train_metadata": metadata.get("manifest_sha256"),
        "manifest_sidecar_sha256": sidecar.read_text().strip(),
        "train_records_deserialized": EXPECTED_TRAIN_COUNT,
        "validation_test_records_deserialized": 0,
        "validation_test_task_payload_loaded": False,
        "validation_test_databases_opened": False,
        "train_task_ids": task_ids,
        "train_database_ids": databases,
        "train_template_ids": templates,
        "dataset_sha256": sha256_file(dataset_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    return cases, boundary


def compare_group_matrix() -> dict[str, Any]:
    fields = ("retain_initial_observation", "canonical_history", "format_retries")
    differences: dict[str, dict[str, list[str]]] = {}
    for left in GROUPS:
        for right in GROUPS:
            if left.name >= right.name:
                continue
            changed = [field for field in fields if getattr(left, field) != getattr(right, field)]
            differences[f"{left.name}__vs__{right.name}"] = {
                "changed_fields": changed,
                "left": left.model_dump(),
                "right": right.model_dump(),
            }
    expected = {
        "baseline__vs__canonical-history": ["canonical_history"],
        "baseline__vs__initial-observation": ["retain_initial_observation"],
        "baseline__vs__no-format-retry": ["format_retries"],
        "baseline__vs__full-alignment": [
            "retain_initial_observation",
            "canonical_history",
            "format_retries",
        ],
    }
    for key, changed in expected.items():
        actual = differences.get(key)
        if actual is None or actual["changed_fields"] != changed:
            raise AssertionError(f"protocol_matrix_unexpected:{key}")
    return {"allowed_factors": list(fields), "pairwise": differences}


def _observation_json(observation: Observation) -> str:
    # This is the exact serialization used by the current LocalModelPolicy.
    return observation.model_dump_json()


def _accepted_history_content(step: StepRecord, canonical_history: bool) -> tuple[str, str]:
    if canonical_history:
        return canonical_json(step.action.model_dump()), "canonical_json"
    if step.model_attempts:
        raw = step.model_attempts[-1].get("raw_output")
        if isinstance(raw, str) and raw:
            return raw, "raw_output"
    return step.action.model_dump_json(), "pydantic_json_fallback"


def build_request_messages(
    *,
    task: PublicTask,
    initial_observation: Observation,
    current_observation: Observation,
    history: list[StepRecord],
    group: ProtocolGroup,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Build one Runtime request and describe accepted-history serialization."""

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    history_modes: list[dict[str, Any]] = []
    if group.retain_initial_observation or not history:
        messages.append({"role": "user", "content": _observation_json(initial_observation)})
    for step in history:
        content, mode = _accepted_history_content(step, group.canonical_history)
        history_modes.append({"step_index": step.step_index, "mode": mode})
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": step.tool_result.model_dump_json()})
    if not history and group.retain_initial_observation is False:
        # The branch above already placed the reset observation.  This explicit
        # condition documents that A uses the current observation only at turn 0.
        pass
    return messages, history_modes


def _public_message_guard(
    messages: list[dict[str, str]],
    *,
    verifier_task: BenchmarkTask | None = None,
) -> None:
    for message in messages:
        if message.get("role") not in {"system", "user"}:
            continue
        content = message.get("content", "")
        if "reference_sql" in content or "expected_answer" in content:
            raise AssertionError("verifier_field_leaked_to_model_request")
        if verifier_task is not None and verifier_task.reference_sql in content:
            raise AssertionError("reference_sql_leaked_to_model_request")


class ProtocolAblationPolicy:
    """Diagnostic policy wrapper with production-compatible request semantics."""

    kind = "local_llm"

    def __init__(
        self,
        backend: GenerationBackend,
        group: ProtocolGroup,
        *,
        model: str = MODEL_ID,
        generation: GenerationSettings | None = None,
        reference_sql_guard: str | None = None,
    ):
        self.backend = backend
        self.group = group
        self.model = model
        base = generation or GenerationSettings()
        self.generation = base.model_copy(update={"format_retries": group.format_retries})
        self.version = f"local-llm-runtime-ablation-{group.name}"
        self.reference_sql_guard = reference_sql_guard
        self.request_log: list[dict[str, Any]] = []
        self._request_index = 0
        self._initial_observation: Observation | None = None

    def _record_request(
        self,
        *,
        task: PublicTask,
        history: list[StepRecord],
        messages: list[dict[str, str]],
        attempt_index: int,
        timeout_s: float,
        response: dict[str, Any] | None,
        raw_output: str | None,
        error: str | None,
        usage: dict[str, Any],
        telemetry: dict[str, Any],
        history_modes: list[dict[str, Any]],
    ) -> None:
        _public_message_guard(messages, verifier_task=None)
        if self.reference_sql_guard is not None:
            for message in messages:
                if message.get("role") in {
                    "system",
                    "user",
                } and self.reference_sql_guard in message.get("content", ""):
                    raise AssertionError("reference_sql_leaked_to_model_request")
        self.request_log.append(
            {
                "request_index": self._request_index,
                "task_id": task.task_id,
                "step_index": len(history),
                "attempt_index": attempt_index,
                "messages": [dict(message) for message in messages],
                "initial_observation_present": any(
                    message.get("content") == _observation_json(self._initial_observation)
                    for message in messages
                    if self._initial_observation is not None
                ),
                "history_assistant_serialization": history_modes,
                "generation": {
                    "temperature": self.generation.temperature,
                    "top_p": self.generation.top_p,
                    "max_tokens": self.generation.max_tokens,
                    "seed": self.generation.seed,
                    "parse_mode": self.generation.parse_mode,
                    "format_retries": self.generation.format_retries,
                    "timeout_s": timeout_s,
                },
                "raw_output": raw_output,
                "error": error,
                "usage": usage,
                "telemetry": telemetry,
                "response_keys": sorted(response) if isinstance(response, dict) else [],
            }
        )
        self._request_index += 1

    def decide(
        self,
        task: PublicTask,
        observation: Observation,
        history: list[StepRecord],
        timeout_s: float,
    ) -> Any:
        if self._initial_observation is None:
            self._initial_observation = observation
        messages, history_modes = build_request_messages(
            task=task,
            initial_observation=self._initial_observation,
            current_observation=observation,
            history=history,
            group=self.group,
        )
        attempts: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout_s
        for retry in range(self.generation.format_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelFailure("inference_timeout", attempts)
            request = BackendRequest(
                model=self.model,
                messages=[dict(message) for message in messages],
                temperature=self.generation.temperature,
                top_p=self.generation.top_p,
                max_tokens=self.generation.max_tokens,
                seed=self.generation.seed,
                timeout_s=remaining,
            )
            started = time.monotonic()
            attempt: dict[str, Any] = {
                "attempt_index": retry,
                "raw_output": None,
                "format_valid": False,
                "strict_format_valid": False,
                "error": None,
                "usage": {},
                "step_index": len(history),
            }
            response: dict[str, Any] | None = None
            try:
                response = self.backend.generate(request)
                if not isinstance(response, dict):
                    raise TypeError("model_response_not_object")
                response_error = response.get("error")
                raw_output = response.get("raw_output")
                usage = response.get("usage") or {}
                telemetry = response.get("telemetry") or {}
                if response_error is not None:
                    code = (
                        response_error.get("code", "model_http_error")
                        if isinstance(response_error, dict)
                        else "model_http_error"
                    )
                    attempt.update(
                        raw_output=raw_output,
                        usage=usage,
                        telemetry=telemetry,
                        error=code,
                        elapsed_s=time.monotonic() - started,
                        server_error=response_error,
                    )
                    attempts.append(attempt)
                    self._record_request(
                        task=task,
                        history=history,
                        messages=messages,
                        attempt_index=retry,
                        timeout_s=remaining,
                        response=response,
                        raw_output=raw_output if isinstance(raw_output, str) else None,
                        error=code,
                        usage=usage,
                        telemetry=telemetry,
                        history_modes=history_modes,
                    )
                    raise ModelFailure(code, attempts)
                if not isinstance(raw_output, str):
                    raise TypeError("model_response_content_not_string")
                attempt.update(raw_output=raw_output, usage=usage, telemetry=telemetry)
            except ModelFailure:
                raise
            except Exception as exc:
                code = (
                    "inference_timeout"
                    if isinstance(exc, TimeoutError)
                    else "model_transport_error"
                )
                attempt.update(
                    error=code,
                    elapsed_s=time.monotonic() - started,
                    exception={"type": type(exc).__name__, "message": str(exc)[:500]},
                )
                attempts.append(attempt)
                self._record_request(
                    task=task,
                    history=history,
                    messages=messages,
                    attempt_index=retry,
                    timeout_s=remaining,
                    response=response,
                    raw_output=None,
                    error=code,
                    usage={},
                    telemetry={},
                    history_modes=history_modes,
                )
                raise ModelFailure(code, attempts) from exc

            raw_output = attempt["raw_output"]
            try:
                action, normalized = parse_action(raw_output, self.generation.parse_mode)
                attempt.update(
                    format_valid=True,
                    strict_format_valid=not normalized,
                    normalized_json_fence=normalized,
                    parsed_action=action.model_dump(),
                )
            except ValueError as exc:
                attempt["error"] = str(exc)
            attempt["elapsed_s"] = time.monotonic() - started
            attempts.append(attempt)
            self._record_request(
                task=task,
                history=history,
                messages=messages,
                attempt_index=retry,
                timeout_s=remaining,
                response=response,
                raw_output=raw_output,
                error=attempt.get("error"),
                usage=attempt.get("usage", {}),
                telemetry=attempt.get("telemetry", {}),
                history_modes=history_modes,
            )
            if attempt["format_valid"]:
                usages = [item.get("usage", {}) for item in attempts]
                return _decision(
                    action=action,
                    attempts=attempts,
                    prompt_tokens=sum(int(item.get("prompt_tokens", 0)) for item in usages),
                    completion_tokens=sum(int(item.get("completion_tokens", 0)) for item in usages),
                    token_usage_known=all(
                        "prompt_tokens" in item and "completion_tokens" in item for item in usages
                    ),
                )
            messages.extend(
                [
                    {"role": "assistant", "content": raw_output},
                    {
                        "role": "user",
                        "content": (
                            "Format error: "
                            + str(attempt["error"])
                            + ". Return one valid JSON Action with documented keys and argument "
                            "types. Do not add commentary or change the business task."
                        ),
                    },
                ]
            )
        raise ModelFailure(attempts[-1].get("error", "invalid_action"), attempts)


def _decision(
    *,
    action: Action,
    attempts: list[dict[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
    token_usage_known: bool,
) -> Any:
    # Importing Decision here keeps the module's public CPU test surface small.
    from enterprise_agent_rl.common import Decision

    return Decision(
        action=action,
        model_attempts=attempts,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        token_usage_known=token_usage_known,
    )


class QwenBackend:
    """Adapter from diagnostic requests to the existing pinned QwenEngine."""

    def __init__(self, engine: Any):
        from enterprise_agent_rl.model_server import CompletionRequest, Message

        self.engine = engine
        self.request_type = CompletionRequest
        self.message_type = Message

    def generate(self, request: BackendRequest) -> dict[str, Any]:
        completion = self.request_type(
            model=request.model,
            messages=[self.message_type.model_validate(message) for message in request.messages],
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            seed=request.seed,
            timeout_s=request.timeout_s,
            stream=False,
        )
        response = self.engine.generate(completion)
        if "raw_output" not in response and response.get("choices"):
            response = dict(response)
            response["raw_output"] = response["choices"][0]["message"].get("content")
        return response


class SequenceMockBackend:
    """Deterministic raw-output backend used only by CPU preflight."""

    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> dict[str, Any]:
        self.calls.append(request)
        index = len(self.calls) - 1
        if index >= len(self.outputs):
            return {"error": {"code": "mock_output_exhausted"}}
        return {
            "raw_output": self.outputs[index],
            "usage": {"prompt_tokens": 10 + index, "completion_tokens": 5},
            "telemetry": {"generation_latency_s": 0.001},
        }


def mock_outputs_for_schema_task() -> list[str]:
    return [
        canonical_json({"kind": "plan", "arguments": {"steps": ["schema", "sql", "check"]}}),
        canonical_json({"kind": "schema", "arguments": {}}),
        canonical_json({"kind": "dictionary", "arguments": {}}),
        canonical_json(
            {
                "kind": "sql",
                "arguments": {
                    "query": "SELECT name AS customer,region_id FROM customers ORDER BY customer"
                },
            }
        ),
        canonical_json(
            {
                "kind": "check",
                "arguments": {"evidence_step": 3, "expected_columns": ["customer", "region_id"]},
            }
        ),
        canonical_json(
            {
                "kind": "finish",
                "arguments": {
                    "evidence_step": 3,
                    "summary": "已根据实际 SQL 执行结果完成查询。",
                },
            }
        ),
    ]


def _task_result(
    case: TrainCase,
    trajectory: Trajectory,
    request_log: list[dict[str, Any]],
) -> dict[str, Any]:
    sql_steps = [step for step in trajectory.steps if step.action.kind == "sql"]
    executable_sql = [step for step in sql_steps if step.tool_result.ok]
    sql_errors = [step for step in sql_steps if not step.tool_result.ok]
    first_sql = min((step.step_index for step in sql_steps), default=None)
    has_schema_or_dictionary = any(
        step.action.kind in {"schema", "dictionary"}
        and (first_sql is None or step.step_index < first_sql)
        for step in trajectory.steps
    )
    query_strings = [step.action.arguments.get("query") for step in sql_steps]
    repeated_sql = len(query_strings) - len(set(query_strings))
    recovery_next_actions: list[str | None] = []
    for error_step in sql_errors:
        next_step = next(
            (step for step in trajectory.steps if step.step_index > error_step.step_index), None
        )
        recovery_next_actions.append(next_step.action.kind if next_step else None)
    attempts = trajectory.model_attempts
    action_kinds = Counter(step.action.kind for step in trajectory.steps)
    parser_errors = Counter(
        str(attempt.get("error"))
        for attempt in attempts
        if attempt.get("error") in {"invalid_json", "invalid_action", "invalid_action_arguments"}
    )
    total_prompt = sum(
        int(attempt.get("usage", {}).get("prompt_tokens", 0)) for attempt in attempts
    )
    total_completion = sum(
        int(attempt.get("usage", {}).get("completion_tokens", 0)) for attempt in attempts
    )
    first_attempt_valid = bool(
        trajectory.steps
        and trajectory.steps[0].model_attempts
        and trajectory.steps[0].model_attempts[0].get("format_valid")
    )
    initial_requests = [item for item in request_log if item.get("initial_observation_present")]
    history_modes = Counter(
        entry.get("mode")
        for item in request_log
        for entry in item.get("history_assistant_serialization", [])
    )
    reward = trajectory.reward.model_dump() if trajectory.reward else None
    failure_reason = reward.get("reason") if reward and not reward.get("success") else None
    return {
        "task_id": case.public_task.task_id,
        "database_id": case.database_id,
        "template_id": case.template_id,
        "teacher_provenance": case.teacher_provenance,
        "success": bool(reward and reward.get("success")),
        "failure_reason": failure_reason or trajectory.terminal_reason,
        "terminal_reason": trajectory.terminal_reason,
        "verifier_result": reward,
        "reward": reward.get("value") if reward else None,
        "latency_s": trajectory.elapsed_s,
        "input_tokens": total_prompt,
        "output_tokens": total_completion,
        "action_count": len(trajectory.steps),
        "tool_calls": sum(step.tool_name is not None for step in trajectory.steps),
        "parser_attempts": len(attempts),
        "parser_acceptances": sum(bool(attempt.get("format_valid")) for attempt in attempts),
        "invalid_json": parser_errors.get("invalid_json", 0),
        "invalid_envelope": parser_errors.get("invalid_action", 0),
        "invalid_arguments": parser_errors.get("invalid_action_arguments", 0),
        "first_action_legal": first_attempt_valid,
        "action_kind_counts": dict(sorted(action_kinds.items())),
        "schema_dictionary_before_first_sql": has_schema_or_dictionary,
        "sql_attempts": len(sql_steps),
        "executable_sql_count": len(executable_sql),
        "sql_error_count": len(sql_errors),
        "sql_error_types": Counter(str(step.tool_result.error) for step in sql_errors),
        "recovery_count": len(sql_errors),
        "recovery_success_count": sum(
            any(
                step.tool_result.ok
                for step in trajectory.steps
                if step.step_index > error.step_index
            )
            for error in sql_errors
        ),
        "error_feedback_next_actions": recovery_next_actions,
        "repeated_sql": repeated_sql,
        "check_present": "check" in action_kinds,
        "finish_present": "finish" in action_kinds,
        "budget_exhausted": trajectory.terminal_reason
        in {"action_budget_exhausted", "tool_budget_exhausted"},
        "format_retry_count": sum(int(attempt.get("attempt_index", 0)) for attempt in attempts),
        "initial_observation_request_count": len(initial_requests),
        "initial_observation_present_in_all_requests": bool(request_log)
        and all(item.get("initial_observation_present") for item in request_log),
        "history_assistant_serialization": dict(sorted(history_modes.items())),
        "request_count": len(request_log),
        "sql_query_sha256": [
            hashlib.sha256(str(query).encode("utf-8")).hexdigest() for query in query_strings
        ],
    }


def _aggregate(records: list[dict[str, Any]], *, include_provenance: bool = True) -> dict[str, Any]:
    count = len(records)
    success_count = sum(bool(item["success"]) for item in records)
    parser_attempts = sum(item["parser_attempts"] for item in records)
    parser_acceptances = sum(item["parser_acceptances"] for item in records)
    first_sql_denominator = count
    sql_attempts = sum(item["sql_attempts"] for item in records)
    executable_sql = sum(item["executable_sql_count"] for item in records)
    sql_errors = sum(item["sql_error_count"] for item in records)
    recovery_count = sum(item["recovery_count"] for item in records)
    recovery_success = sum(item["recovery_success_count"] for item in records)
    action_kinds: Counter[str] = Counter()
    sql_error_types: Counter[str] = Counter()
    provenance_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        action_kinds.update(item["action_kind_counts"])
        sql_error_types.update(item["sql_error_types"])
        provenance_groups[item["teacher_provenance"]].append(item)

    def ratio(numerator: int | float, denominator: int | float) -> float:
        return numerator / denominator if denominator else 0.0

    def average(key: str) -> float:
        return sum(float(item[key] or 0) for item in records) / count if count else 0.0

    result = {
        "task_count": count,
        "task_success_rate": ratio(success_count, count),
        "task_success_count": success_count,
        "parser_acceptance_rate": ratio(parser_acceptances, parser_attempts),
        "parser_acceptances": parser_acceptances,
        "parser_attempts": parser_attempts,
        "first_action_legal_rate": ratio(
            sum(item["first_action_legal"] for item in records), count
        ),
        "action_kind_distribution": dict(sorted(action_kinds.items())),
        "schema_dictionary_before_first_sql_coverage": ratio(
            sum(item["schema_dictionary_before_first_sql"] for item in records),
            first_sql_denominator,
        ),
        "sql_executable_rate": ratio(executable_sql, sql_attempts),
        "sql_error_rate": ratio(sql_errors, sql_attempts),
        "recovery_success_rate": ratio(recovery_success, recovery_count),
        "sql_repeat_rate": ratio(sum(item["repeated_sql"] for item in records), sql_attempts),
        "check_rate": ratio(sum(item["check_present"] for item in records), count),
        "finish_rate": ratio(sum(item["finish_present"] for item in records), count),
        "budget_exhaustion_rate": ratio(sum(item["budget_exhausted"] for item in records), count),
        "average_tool_calls": average("tool_calls"),
        "average_latency_s": average("latency_s"),
        "average_input_tokens": average("input_tokens"),
        "average_output_tokens": average("output_tokens"),
        "average_total_tokens": average("input_tokens") + average("output_tokens"),
        "average_reward": average("reward"),
        "invalid_json_count": sum(item["invalid_json"] for item in records),
        "invalid_envelope_count": sum(item["invalid_envelope"] for item in records),
        "invalid_arguments_count": sum(item["invalid_arguments"] for item in records),
        "format_retry_count": sum(item["format_retry_count"] for item in records),
        "sql_error_types": dict(sorted(sql_error_types.items())),
    }
    if include_provenance:
        result["teacher_provenance"] = {
            key: _aggregate(value, include_provenance=False)
            for key, value in sorted(provenance_groups.items())
        }
    return result


def _per_template(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["template_id"]].append(record)
    return {template: _aggregate(items) for template, items in sorted(grouped.items())}


def run_group(
    *,
    group: ProtocolGroup,
    cases: list[TrainCase],
    backend: GenerationBackend,
    settings: Settings,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    request_path = output_dir / "model-requests.jsonl"
    raw_path = output_dir / "raw-generations.jsonl"
    sql_path = output_dir / "sql-audit.jsonl"
    results: list[dict[str, Any]] = []
    for case in cases:
        policy = ProtocolAblationPolicy(
            backend,
            group,
            generation=GenerationSettings(
                temperature=0,
                top_p=1,
                max_tokens=512,
                format_retries=group.format_retries,
                parse_mode=PARSER_MODE,
                seed=17,
            ),
            reference_sql_guard=case.verifier_task.reference_sql,
        )
        trajectory = run_episode(
            case.public_task,
            policy,
            case.database,
            settings,
            audit_path=sql_path,
        )
        reward = verify(case.verifier_task, trajectory, case.database, settings)
        trajectory = trajectory.model_copy(
            update={
                "reward": reward,
                "policy_metadata": {
                    "protocol_group": group.model_dump(),
                    "generation": policy.generation.model_dump(),
                    "train_only_loader": "stream_first_train_records_only",
                },
            }
        )
        task_requests = [dict(item) for item in policy.request_log]
        for item in task_requests:
            item["protocol_group"] = group.name
            item["database_id"] = case.database_id
            _public_message_guard(item["messages"])
            append_jsonl(request_path, item)
        for attempt in trajectory.model_attempts:
            append_jsonl(
                raw_path,
                {
                    "task_id": case.public_task.task_id,
                    "database_id": case.database_id,
                    "template_id": case.template_id,
                    "protocol_group": group.name,
                    **attempt,
                },
            )
        result = _task_result(case, trajectory, task_requests)
        write_json(
            output_dir / f"task-{case.public_task.task_id}-trajectory.json",
            trajectory.model_dump(),
        )
        write_json(output_dir / f"task-{case.public_task.task_id}-result.json", result)
        results.append(result)
    summary = {
        "schema_version": "1.0",
        "group": group.model_dump(),
        "task_count": len(results),
        "complete": len(results) == len(cases),
        "results": results,
        "metrics": _aggregate(results),
        "per_template": _per_template(results),
        "files": {
            "model_requests": str(request_path),
            "raw_generations": str(raw_path),
            "sql_audit": str(sql_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _assert_no_reference_in_public_artifacts(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "reference_sql" in text or "expected_answer" in text:
        raise AssertionError(f"verifier_payload_leaked_to_public_artifact:{path.name}")


def run_cpu_mock_checks(
    *,
    cases: list[TrainCase],
    settings: Settings,
    output_dir: Path,
) -> dict[str, Any]:
    schema_case = next(case for case in cases if case.template_id == "schema")
    mock_summaries: dict[str, Any] = {}
    for group in GROUPS:
        backend = SequenceMockBackend(mock_outputs_for_schema_task())
        summary = run_group(
            group=group,
            cases=[schema_case],
            backend=backend,
            settings=settings,
            output_dir=output_dir / group.name,
        )
        if not summary["complete"] or summary["metrics"]["task_success_rate"] != 1.0:
            raise AssertionError(f"mock_episode_failed:{group.name}")
        trajectory_path = (
            output_dir / group.name / f"task-{schema_case.public_task.task_id}-trajectory.json"
        )
        _assert_no_reference_in_public_artifacts(trajectory_path)
        mock_summaries[group.name] = {
            "backend_calls": len(backend.calls),
            "summary": summary,
        }
    return {
        "status": "passed",
        "groups": mock_summaries,
        "model_weights_loaded": False,
        "gpu_used": False,
        "reference_sql_in_model_requests": False,
    }


def run_protocol_self_checks(cases: list[TrainCase], settings: Settings) -> dict[str, Any]:
    matrix = compare_group_matrix()
    case = next(case for case in cases if case.template_id == "schema")
    initial = Observation(
        step_index=-1,
        ok=True,
        result={"task": case.public_task.model_dump(), "seed": settings.seed},
    )
    first_action = Action(kind="schema", arguments={})
    first_step = StepRecord(
        step_index=0,
        observation=initial,
        action=first_action,
        model_attempts=[
            {
                "raw_output": '{"kind":"schema","arguments":{}}',
                "format_valid": True,
            }
        ],
        tool_name="schema",
        tool_arguments={},
        tool_result=Observation(step_index=0, ok=True, result={"schema": []}),
        prompt_tokens=1,
        completion_tokens=1,
        token_usage_known=True,
        elapsed_s=0,
    )
    messages_a, modes_a = build_request_messages(
        task=case.public_task,
        initial_observation=initial,
        current_observation=initial,
        history=[],
        group=GROUPS[0],
    )
    messages_b, _ = build_request_messages(
        task=case.public_task,
        initial_observation=initial,
        current_observation=first_step.tool_result,
        history=[first_step],
        group=GROUPS[1],
    )
    messages_c, modes_c = build_request_messages(
        task=case.public_task,
        initial_observation=initial,
        current_observation=first_step.tool_result,
        history=[first_step],
        group=GROUPS[2],
    )
    initial_text = _observation_json(initial)
    if messages_a[2]["content"] != initial_text:
        raise AssertionError("initial_observation_first_turn_position_failed")
    if messages_b[2]["content"] != initial_text or messages_b[3]["role"] != "assistant":
        raise AssertionError("initial_observation_retention_position_failed")
    if sum(message.get("content") == initial_text for message in messages_b) != 1:
        raise AssertionError("initial_observation_duplicate_failed")
    if modes_a or modes_c != [{"step_index": 0, "mode": "canonical_json"}]:
        raise AssertionError("history_serializer_mode_failed")
    if messages_c[2]["content"] != canonical_json(first_action.model_dump()):
        raise AssertionError("canonical_serializer_mismatch")

    invalid_backend = SequenceMockBackend(
        ["not json", canonical_json({"kind": "schema", "arguments": {}})]
    )
    retry_policy = ProtocolAblationPolicy(invalid_backend, GROUPS[3])
    try:
        retry_policy.decide(case.public_task, initial, [], 10)
    except ModelFailure:
        pass
    else:
        raise AssertionError("format_retry_zero_did_not_fail_closed")
    if len(invalid_backend.calls) != 1 or len(retry_policy.request_log) != 1:
        raise AssertionError("format_retry_zero_called_twice")

    retry_backend = SequenceMockBackend(
        ["not json", canonical_json({"kind": "schema", "arguments": {}})]
    )
    retry_policy_one = ProtocolAblationPolicy(retry_backend, GROUPS[0])
    decision = retry_policy_one.decide(case.public_task, initial, [], 10)
    if decision.action.kind != "schema" or len(retry_backend.calls) != 2:
        raise AssertionError("format_retry_one_failed")
    for request in retry_policy_one.request_log:
        _public_message_guard(request["messages"], verifier_task=case.verifier_task)

    return {
        "status": "passed",
        "protocol_matrix": matrix,
        "initial_observation_position": "passed",
        "initial_observation_unique": "passed",
        "canonical_serializer": "passed",
        "format_retries_zero": "passed",
        "format_retries_one": "passed",
        "reference_payload_request_guard": "passed",
    }


def _effective_settings(path: Path) -> tuple[Settings, dict[str, Any]]:
    settings = Settings.load(path)
    requested = {
        "seed": 17,
        "max_actions": 20,
        "max_tool_calls": 14,
        "episode_timeout_s": 600.0,
        "inference_timeout_s": 120.0,
        "sql_timeout_s": 2.0,
        "sql_max_rows": 100,
        "sql_max_bytes": 65536,
        "sql_memory_mb": 256,
    }
    effective = settings.model_dump()
    differences = {
        key: {"requested": value, "effective": effective.get(key)}
        for key, value in requested.items()
        if effective.get(key) != value
    }
    return settings, {"requested": requested, "effective": effective, "differences": differences}


def _generation_config(path: Path) -> dict[str, Any]:
    import tomllib

    values = tomllib.loads(path.read_text())
    generation = values.get("generation", {})
    required = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "format_retries": 1,
        "parse_mode": PARSER_MODE,
        "seed": 17,
    }
    differences = {
        key: {"expected": value, "actual": generation.get(key)}
        for key, value in required.items()
        if generation.get(key) != value
    }
    return {
        "model_id": values.get("model_id"),
        "revision": values.get("revision"),
        "dtype": values.get("dtype"),
        "attention_backend": values.get("attention_backend", values.get("attention")),
        "generation": generation,
        "differences": differences,
    }


def _tokenizer_source_evidence(adapter: Path) -> dict[str, Any]:
    config_path = adapter / "tokenizer_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text())
    template = config.get("chat_template")
    if not template:
        raise ValueError("adapter_chat_template_missing")
    template_bytes = canonical_json(template).encode("utf-8")
    tokenizer_files = [
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    hashes = {
        name: sha256_file(adapter / name) for name in tokenizer_files if (adapter / name).is_file()
    }
    return {
        "source": "sft_adapter_tokenizer_files",
        "tokenizer_files": hashes,
        "chat_template_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "chat_template_bytes": len(template_bytes),
        "model_weights_loaded": False,
    }


def _cache_evidence(cache: Path) -> dict[str, Any]:
    manifest_path = cache / "qwen-cache-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("model_cache_identity_mismatch")
    incomplete = sorted(str(path) for path in cache.rglob("*.incomplete"))
    if incomplete:
        raise ValueError("model_cache_incomplete")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model": manifest.get("model"),
        "revision": manifest.get("revision"),
        "weight_files": manifest.get("weight_files"),
        "incomplete_files": incomplete,
        "snapshot": manifest.get("snapshot"),
        "model_weights_loaded": False,
    }


def build_preflight(
    *,
    root: Path,
    run_dir: Path,
    dataset: Path,
    metadata: Path,
    manifest: Path,
    database_root: Path,
    adapter: Path,
    cache: Path,
    agent_config: Path,
    model_config: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings, settings_evidence = _effective_settings(agent_config)
    model_evidence = _generation_config(model_config)
    if model_evidence["model_id"] != MODEL_ID or model_evidence["revision"] != MODEL_REVISION:
        raise ValueError("pinned_model_config_mismatch")
    if model_evidence["differences"]:
        raise ValueError(f"generation_config_mismatch:{model_evidence['differences']}")
    adapter_weight = adapter / "adapter_model.safetensors"
    adapter_sha = sha256_file(adapter_weight)
    if adapter_sha != EXPECTED_ADAPTER_SHA256:
        raise ValueError("adapter_sha256_mismatch")
    cases, boundary = load_train_only_cases(
        manifest_path=manifest,
        dataset_path=dataset,
        metadata_path=metadata,
        database_root=database_root,
    )
    self_checks = run_protocol_self_checks(cases, settings)
    mock_checks = run_cpu_mock_checks(
        cases=cases, settings=settings, output_dir=run_dir / "cpu-mock"
    )
    tokenizer = _tokenizer_source_evidence(adapter)
    cache_evidence = _cache_evidence(cache)
    git = git_state()
    preflight = {
        "schema_version": "1.0",
        "status": "READY_FOR_GPU_SUBMISSION",
        "run_id": run_dir.name,
        "scope": {
            "split": "train",
            "task_count": len(cases),
            "database_ids": sorted({case.database_id for case in cases}),
            "template_ids": sorted({case.template_id for case in cases}),
            "validation_benchmark_run": False,
            "test_split_accessed": False,
            "training_or_grpo": False,
        },
        "data_boundary": boundary,
        "adapter": {
            "path": str(adapter),
            "weight_sha256": adapter_sha,
            "expected_sha256": EXPECTED_ADAPTER_SHA256,
        },
        "model": model_evidence,
        "settings": settings_evidence,
        "tokenizer": tokenizer,
        "cache": cache_evidence,
        "protocol_groups": [group.model_dump() for group in GROUPS],
        "protocol_matrix": self_checks["protocol_matrix"],
        "checks": {
            "train_only_loader": "passed",
            "only_three_protocol_factors_vary": "passed",
            "initial_observation_position_and_uniqueness": "passed",
            "canonical_serializer_matches_sft": "passed",
            "format_retries_zero_no_second_call": "passed",
            "reference_sql_expected_answer_not_in_requests": "passed",
            "mock_complete_episode_for_all_groups": "passed",
            "git_diff_check": "pending_external_check",
        },
        "cpu_preflight": {
            "model_weights_loaded": False,
            "gpu_used": False,
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "self_checks": self_checks,
            "mock_checks": mock_checks,
        },
        "git": git,
    }
    environment = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "phase": "cpu_preflight_complete_waiting_gpu",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "job_id": None,
        "node": platform.node(),
        "partition": None,
        "exit_code": None,
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "adapter_sha256": adapter_sha,
            "dtype": model_evidence["dtype"],
            "attention_backend": model_evidence["attention_backend"],
        },
        "tokenizer": tokenizer,
        "git_at_preflight": git,
        "gpu": None,
        "peak_gpu_allocated_bytes": None,
        "peak_gpu_reserved_bytes": None,
    }
    return preflight, environment


def _scheduler_command_results(partition: str, account: str, qos: str) -> dict[str, Any]:
    return {
        "sinfo": command_result(
            [
                "sinfo",
                "-h",
                "-p",
                partition,
                "-o",
                "%P|%a|%l|%G|%D|%F|%C",
            ]
        ),
        "sinfo_nodes": command_result(
            ["sinfo", "-h", "-N", "-p", partition, "-o", "%P|%N|%T|%G|%c|%m"]
        ),
        "squeue": command_result(["squeue", "-h", "-p", partition, "-o", "%i|%T|%R|%M|%l|%P|%u"]),
        "sacct_recent": command_result(
            [
                "sacct",
                "-X",
                "-S",
                "now-7days",
                "-A",
                account,
                "-o",
                "JobIDRaw,Partition,State,ElapsedRaw,Start,End",
                "-n",
                "-P",
            ]
        ),
        "sacctmgr_assoc": command_result(
            [
                "sacctmgr",
                "-nP",
                "show",
                "assoc",
                "where",
                f"account={account}",
                "format=Account,Partition,QOS,User,DefaultQOS",
            ]
        ),
        "sacctmgr_qos": command_result(
            [
                "sacctmgr",
                "-nP",
                "show",
                "qos",
                qos,
                "format=Name,MaxWall,GrpTRES,MaxTRESPU,MaxJobsPU",
            ]
        ),
        "sbatch_test_only": command_result(
            [
                "sbatch",
                "--test-only",
                "--account",
                account,
                "--partition",
                partition,
                "--qos",
                qos,
                "--nodes",
                "1",
                "--ntasks",
                "1",
                "--cpus-per-task",
                "2",
                "--gres",
                "gpu:tesla:1",
                "--mem",
                "16G",
                "--time",
                "01:00:00",
                "--wrap",
                "true",
            ]
        ),
    }


def _partition_looks_compatible(evidence: dict[str, Any]) -> bool:
    text = evidence["sinfo_nodes"]["stdout"].lower()
    return "v100" in text or "tesla" in text or "gpu:tesla" in text


def _historical_v100_evidence(root: Path, partition: str) -> dict[str, Any]:
    candidates = {
        "gpu2Q": root / "artifacts/gpu-inventory-20385803/inventory.json",
        "gpu8Q": root / "artifacts/base-agent-smoke-20386884/environment.json",
    }
    path = candidates.get(partition)
    if path is None or not path.is_file():
        return {"known_v100": False, "path": str(path) if path else None}
    text = path.read_text(encoding="utf-8")
    compute_capability_70 = "7.0" in text
    try:
        value = json.loads(text)

        def contains_cc70(item: Any) -> bool:
            if isinstance(item, dict):
                return any(
                    key in {"compute_capability", "cc"}
                    and item[key] == [7, 0]
                    or contains_cc70(child)
                    for key, child in item.items()
                )
            if isinstance(item, list):
                return any(contains_cc70(child) for child in item)
            return False

        compute_capability_70 |= contains_cc70(value)
    except json.JSONDecodeError:
        pass
    known = "Tesla V100" in text and compute_capability_70
    return {"known_v100": known, "path": str(path)}


def _predicted_start(evidence: dict[str, Any]) -> str | None:
    text = evidence["sbatch_test_only"].get("stderr", "")
    match = re.search(r"to start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", text)
    return match.group(1) if match else None


def select_partition(
    *,
    run_dir: Path,
    root: Path,
    account: str = "pi_daipeishan",
    qos: str = "gpuq",
) -> str:
    preflight_path = run_dir / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "READY_FOR_GPU_SUBMISSION":
        raise RuntimeError("cpu_preflight_not_ready")
    evidence: dict[str, Any] = {
        "account": account,
        "qos": qos,
        "checked_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "partitions": {},
        "selection_order": ["gpu2Q", "gpu4Q", "gpu8Q"],
    }
    selected = None
    reasons: list[str] = []
    for partition in evidence["selection_order"]:
        part_evidence = _scheduler_command_results(partition, account, qos)
        evidence["partitions"][partition] = part_evidence
        test_ok = part_evidence["sbatch_test_only"]["returncode"] == 0
        sinfo_ok = part_evidence["sinfo_nodes"]["returncode"] == 0
        compatible = _partition_looks_compatible(part_evidence)
        queue_lines = [
            line for line in part_evidence["squeue"]["stdout"].splitlines() if line.strip()
        ]
        pending_lines = [line for line in queue_lines if "PEND" in line.upper()]
        part_evidence["queue_summary"] = {
            "job_count": len(queue_lines),
            "pending_count": len(pending_lines),
            "pending_reasons": [line.split("|", 3)[2] for line in pending_lines],
        }
        part_evidence["selection_checks"] = {
            "sbatch_test_only_ok": test_ok,
            "sinfo_ok": sinfo_ok,
            "v100_or_tesla_resource_visible": compatible,
        }
        historical = _historical_v100_evidence(root, partition)
        part_evidence["historical_v100_evidence"] = historical
        predicted = _predicted_start(part_evidence)
        part_evidence["predicted_start"] = predicted
        part_evidence["selection_checks"]["historical_v100_confirmed"] = historical["known_v100"]
        part_evidence["selection_checks"]["compatible_v100_candidate"] = (
            compatible and historical["known_v100"]
        )
        if not (test_ok and sinfo_ok and compatible and historical["known_v100"]):
            reasons.append(
                f"skipped:{partition}:test_only={test_ok}:sinfo={sinfo_ok}:"
                f"gres={compatible}:historical_v100={historical['known_v100']}"
            )

    candidates = [
        partition
        for partition, item in evidence["partitions"].items()
        if item["selection_checks"].get("compatible_v100_candidate")
    ]
    preferred = "gpu2Q" if "gpu2Q" in candidates else None
    predicted = {
        partition: evidence["partitions"][partition].get("predicted_start")
        for partition in candidates
    }
    if preferred and predicted[preferred]:
        preferred_time = dt.datetime.fromisoformat(predicted[preferred])
        other_times = [
            dt.datetime.fromisoformat(value)
            for partition, value in predicted.items()
            if partition != preferred and value
        ]
        if not other_times or preferred_time <= min(other_times) + dt.timedelta(hours=1):
            selected = preferred
            reasons.append(
                f"selected_preferred_partition_within_one_hour:{preferred}:"
                f"predicted_start={predicted[preferred]}"
            )
        else:
            selected = min(
                (partition for partition, value in predicted.items() if value),
                key=lambda partition: predicted[partition],
            )
            reasons.append(
                f"preferred_partition_significantly_later:{preferred}:"
                f"selected_earliest_predicted_start={selected}"
            )
    elif candidates:
        with_start = [partition for partition in candidates if predicted[partition]]
        selected = (
            min(with_start, key=lambda partition: predicted[partition])
            if with_start
            else candidates[0]
        )
        reasons.append(f"selected_first_evidence_compatible_partition:{selected}")
    if selected is None:
        preflight["status"] = "SFT_RUNTIME_ABLATION_BLOCKED"
        preflight["scheduler"] = {"status": "blocked", "evidence": evidence, "reasons": reasons}
        write_json(preflight_path, preflight)
        raise RuntimeError("no_compatible_gpu_partition")
    evidence["selected_partition"] = selected
    evidence["selection_reasons"] = reasons
    preflight["scheduler"] = {"status": "ready", "evidence": evidence}
    write_json(preflight_path, preflight)
    environment_path = run_dir / "environment.json"
    environment = json.loads(environment_path.read_text())
    environment["partition"] = selected
    environment["scheduler"] = preflight["scheduler"]
    write_json(environment_path, environment)
    return selected


def _gpu_environment(run_dir: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    from enterprise_agent_rl.gpu_guard import require_compute_node

    require_compute_node()
    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires_exactly_one_visible_gpu")
    capability = list(torch.cuda.get_device_capability(0))
    name = torch.cuda.get_device_name(0)
    if "V100" not in name.upper() or capability != [7, 0]:
        raise RuntimeError(f"requires_tesla_v100_cc70:{name}:{capability}")
    props = torch.cuda.get_device_properties(0)
    environment = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "account": os.environ.get("SLURM_JOB_ACCOUNT", "pi_daipeishan"),
        "qos": os.environ.get("SLURM_JOB_QOS", "gpuq"),
        "gpu_name": name,
        "compute_capability": capability,
        "device_count": torch.cuda.device_count(),
        "total_memory_bytes": int(props.total_memory),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_arch_list": torch.cuda.get_arch_list(),
        "nvidia_smi": command_result(["nvidia-smi"], timeout=30),
        "scontrol_job": command_result(
            ["scontrol", "show", "job", os.environ.get("SLURM_JOB_ID", "")], timeout=30
        ),
        "single_visible_gpu": True,
        "v100_gate_passed": True,
        "weights_loaded_on_cpu": False,
    }
    return environment


def _actual_tokenizer_identity(engine: Any) -> dict[str, Any]:
    template = engine.tokenizer.chat_template
    template_bytes = canonical_json(template).encode("utf-8")
    vocab = engine.tokenizer.get_vocab()
    vocab_bytes = canonical_json(vocab).encode("utf-8")
    return {
        "tokenizer_class": type(engine.tokenizer).__name__,
        "tokenizer_name_or_path": str(getattr(engine.tokenizer, "name_or_path", "")),
        "tokenizer_vocab_sha256": hashlib.sha256(vocab_bytes).hexdigest(),
        "chat_template_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "chat_template_bytes": len(template_bytes),
        "tokenizer_vocab_size": len(vocab),
    }


def run_gpu(
    *,
    root: Path,
    run_dir: Path,
    dataset: Path,
    metadata: Path,
    manifest: Path,
    database_root: Path,
    adapter: Path,
    cache: Path,
    agent_config: Path,
    model_config: Path,
) -> int:
    preflight = json.loads((run_dir / "preflight.json").read_text())
    if preflight.get("status") != "READY_FOR_GPU_SUBMISSION":
        raise RuntimeError("cpu_preflight_not_ready")
    selected = preflight.get("scheduler", {}).get("evidence", {}).get("selected_partition")
    if selected != os.environ.get("SLURM_JOB_PARTITION"):
        raise RuntimeError("slurm_partition_does_not_match_preflight_selection")
    cases, boundary = load_train_only_cases(
        manifest_path=manifest,
        dataset_path=dataset,
        metadata_path=metadata,
        database_root=database_root,
    )
    current_adapter_sha = sha256_file(adapter / "adapter_model.safetensors")
    if current_adapter_sha != preflight["adapter"]["weight_sha256"]:
        raise RuntimeError("adapter_changed_after_preflight")
    settings, settings_evidence = _effective_settings(agent_config)
    model_evidence = _generation_config(model_config)
    if model_evidence["differences"]:
        raise RuntimeError("generation_config_changed_after_preflight")
    gpu = _gpu_environment(run_dir, preflight)
    environment_path = run_dir / "environment.json"
    environment = json.loads(environment_path.read_text())
    environment.update(
        {
            "phase": "gpu_running",
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": platform.node(),
            "partition": selected,
            "gpu": gpu,
            "git_at_gpu_start": git_state(),
            "data_boundary_at_gpu_start": boundary,
            "settings_at_gpu_start": settings_evidence,
            "adapter_sha256_at_gpu_start": current_adapter_sha,
        }
    )
    write_json(environment_path, environment)

    from enterprise_agent_rl.model_server import ModelSettings, QwenEngine

    cfg = ModelSettings.load(model_config)
    engine = None
    summaries: list[dict[str, Any]] = []
    fatal: dict[str, Any] | None = None
    try:
        engine = QwenEngine(cfg, cache, run_dir / "engine", adapter=adapter)
        identity = _actual_tokenizer_identity(engine)
        expected_chat_hash = preflight["tokenizer"]["chat_template_sha256"]
        if identity["chat_template_sha256"] != expected_chat_hash:
            raise RuntimeError("chat_template_hash_changed_after_preflight")
        environment["model_load"] = engine.metadata
        environment["tokenizer_actual"] = identity
        write_json(environment_path, environment)
        backend = QwenBackend(engine)
        for group in GROUPS:
            summary = run_group(
                group=group,
                cases=cases,
                backend=backend,
                settings=settings,
                output_dir=run_dir / group.name,
            )
            summary["model_revision"] = cfg.revision
            summary["adapter_sha256"] = current_adapter_sha
            summary["tokenizer_chat_template_sha256"] = identity["chat_template_sha256"]
            summary["gpu"] = gpu
            write_json(run_dir / group.name / "summary.json", summary)
            summaries.append(summary)
    except BaseException as exc:
        fatal = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    finally:
        if engine is not None:
            try:
                import torch

                environment["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                environment["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as exc:
                environment["gpu_memory_capture_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            del engine
            gc.collect()
    comparison = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "status": (
            "SFT_RUNTIME_ABLATION_COMPLETE"
            if fatal is None and len(summaries) == len(GROUPS)
            else "SFT_RUNTIME_ABLATION_FAILED"
        ),
        "scope": preflight["scope"],
        "groups": {
            summary["group"]["name"]: {
                "config": summary["group"],
                "metrics": summary["metrics"],
                "per_template": summary["per_template"],
                "task_count": summary["task_count"],
                "complete": summary["complete"],
            }
            for summary in summaries
        },
        "interpretation": interpret_comparison(summaries),
        "completed_group_count": len(summaries),
        "fatal_error": fatal,
        "fixed_conditions": {
            "model_id": MODEL_ID,
            "revision": cfg.revision,
            "adapter_sha256": current_adapter_sha,
            "settings": settings.model_dump(),
            "generation_base": model_evidence["generation"],
            "attention_backend": cfg.attention_backend,
        },
    }
    write_json(run_dir / "comparison.json", comparison)
    environment["phase"] = "gpu_complete" if fatal is None else "gpu_failed"
    environment["completed_group_count"] = len(summaries)
    environment["fatal_error"] = fatal
    write_json(environment_path, environment)
    return 0 if comparison["status"] == "SFT_RUNTIME_ABLATION_COMPLETE" else 2


def interpret_comparison(summaries: list[dict[str, Any]]) -> list[str]:
    by_name = {summary["group"]["name"]: summary["metrics"] for summary in summaries}
    if not by_name:
        return ["No complete group was available for interpretation."]
    observations: list[str] = []
    baseline = by_name.get("baseline")
    if baseline is not None:
        for name, text in (
            ("initial-observation", "supports initial observation retention"),
            ("canonical-history", "supports canonical history serialization"),
        ):
            if (
                name in by_name
                and by_name[name]["task_success_rate"] > baseline["task_success_rate"]
            ):
                observations.append(
                    f"{name} success exceeds baseline: {text}; "
                    "this is diagnostic support, not proof of uniqueness."
                )
        if "no-format-retry" in by_name:
            no_retry = by_name["no-format-retry"]
            if (
                no_retry["invalid_json_count"]
                + no_retry["invalid_envelope_count"]
                + no_retry["invalid_arguments_count"]
                > baseline["invalid_json_count"]
                + baseline["invalid_envelope_count"]
                + baseline["invalid_arguments_count"]
                and no_retry["format_retry_count"] < baseline["format_retry_count"]
            ):
                observations.append(
                    "No-format-retry increases parser failures while removing retry calls; "
                    "this is a retry trade-off, not a parser-only win."
                )
        if "full-alignment" in by_name:
            best_single = max(
                (
                    by_name[name]["task_success_rate"]
                    for name in ("initial-observation", "canonical-history", "no-format-retry")
                    if name in by_name
                ),
                default=-1,
            )
            if by_name["full-alignment"]["task_success_rate"] > best_single:
                observations.append(
                    "Full alignment exceeds every single-factor group, consistent with a "
                    "possible complementary protocol effect."
                )
    if all(metrics["task_success_rate"] == 0 for metrics in by_name.values()):
        observations.append(
            "All groups have zero success; inspect first actions, schema/dictionary coverage, "
            "SQL errors, recovery, and parser failures before concluding protocol irrelevance."
        )
    observations.append(
        "Twenty train tasks are diagnostic evidence only; remaining uncertainty includes data "
        "heterogeneity, SQL argument learning, and autonomous planning."
    )
    return observations


def finalize_job(*, run_dir: Path, exit_code: int) -> None:
    path = run_dir / "environment.json"
    environment = (
        json.loads(path.read_text())
        if path.is_file()
        else {
            "schema_version": "1.0",
            "run_id": run_dir.name,
        }
    )
    environment["exit_code"] = exit_code
    environment["job_id"] = os.environ.get("SLURM_JOB_ID", environment.get("job_id"))
    environment["node"] = platform.node()
    environment["partition"] = os.environ.get("SLURM_JOB_PARTITION", environment.get("partition"))
    environment["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    write_json(path, environment)


def record_diff_check(*, run_dir: Path) -> None:
    preflight_path = run_dir / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "READY_FOR_GPU_SUBMISSION":
        raise RuntimeError("cpu_preflight_not_ready")
    result = command_result(["git", "diff", "--check"])
    if result["returncode"] != 0:
        preflight["status"] = "SFT_RUNTIME_ABLATION_BLOCKED"
        preflight["checks"]["git_diff_check"] = {
            "status": "failed",
            "command": result,
        }
        write_json(preflight_path, preflight)
        raise RuntimeError("git_diff_check_failed")
    preflight["checks"]["git_diff_check"] = {
        "status": "passed",
        "command": result,
    }
    write_json(preflight_path, preflight)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("cpu-preflight", "record-diff-check", "scheduler", "gpu", "finalize"),
        required=True,
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/sft-data-train-teacher-v2/dataset.jsonl")
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/sft-data-train-teacher-v2/dataset-metadata.json"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/m2a/manifest.json"))
    parser.add_argument("--database-root", type=Path, default=Path("artifacts/m2a-databases-final"))
    parser.add_argument("--adapter", type=Path, default=ADAPTER_RELATIVE)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/model-cache"))
    parser.add_argument("--agent-config", type=Path, default=Path("configs/agent/qwen.toml"))
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen2.5-1.5b.toml")
    )
    parser.add_argument("--account", default="pi_daipeishan")
    parser.add_argument("--qos", default="gpuq")
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    run_dir = args.run_dir.resolve()
    paths = {
        key: (root / value if not value.is_absolute() else value).resolve()
        for key, value in {
            "dataset": args.dataset,
            "metadata": args.metadata,
            "manifest": args.manifest,
            "database_root": args.database_root,
            "adapter": args.adapter,
            "cache": args.cache,
            "agent_config": args.agent_config,
            "model_config": args.model_config,
        }.items()
    }
    if args.mode == "finalize":
        finalize_job(run_dir=run_dir, exit_code=args.exit_code)
        return 0
    if args.mode == "record-diff-check":
        record_diff_check(run_dir=run_dir)
        return 0
    if args.mode == "scheduler":
        select_partition(run_dir=run_dir, root=root, account=args.account, qos=args.qos)
        return 0
    if args.mode == "cpu-preflight":
        if run_dir.exists():
            raise SystemExit(f"refusing_to_overwrite_existing_run:{run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            preflight, environment = build_preflight(root=root, run_dir=run_dir, **paths)
        except BaseException as exc:
            preflight = {
                "schema_version": "1.0",
                "status": "SFT_RUNTIME_ABLATION_BLOCKED",
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
                "error": preflight["error"],
                "git_at_preflight": preflight["git"],
            }
            write_json(run_dir / "preflight.json", preflight)
            write_json(run_dir / "environment.json", environment)
            return 2
        write_json(run_dir / "preflight.json", preflight)
        write_json(run_dir / "environment.json", environment)
        return 0
    try:
        return run_gpu(root=root, run_dir=run_dir, **paths)
    except BaseException as exc:
        write_json(
            run_dir / "comparison.json",
            {
                "schema_version": "1.0",
                "run_id": run_dir.name,
                "status": "SFT_RUNTIME_ABLATION_FAILED",
                "fatal_error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
            },
        )
        finalize_job(run_dir=run_dir, exit_code=2)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
