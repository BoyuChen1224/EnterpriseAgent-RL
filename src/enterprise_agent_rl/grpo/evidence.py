"""Immutable model-call evidence and pre-verifier qualification.

The online runner is allowed to resume, but a model call is not allowed to be
resumed by replacing an old request/response file.  This module gives every
call a durable attempt number and unique request ID, writes the intent before
model execution, and binds the response, engine event, token record, Action
attempt, and trajectory before a verifier is allowed to run.
"""

from __future__ import annotations

import copy
import fcntl
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import EpisodeSample, TokenRecord
from .io import (
    ImmutableWriteError,
    append_jsonl,
    canonical_hash,
    exclusive_write_json,
    read_json,
    safe_relative_path,
    sha256_file,
    sha256_text,
)


class EvidenceError(RuntimeError):
    """Base class for evidence persistence and qualification failures."""


class EvidencePersistenceError(EvidenceError):
    """A required immutable evidence artifact could not be persisted."""


class QualificationError(EvidenceError):
    """A model episode cannot be safely sent to the verifier."""


_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(value: Any) -> str:
    text = _SLUG_RE.sub("_", str(value)).strip("_")
    return text or "unknown"


@dataclass(frozen=True)
class AttemptHandle:
    stage: str
    arm: str
    attempt_index: int
    attempt_id: str
    request_id: str
    response_id: str
    logical_request_index: int
    request_path: str
    response_path: str
    engine_event_path: str
    failure_path: str


def _relative(run_dir: Path, path: Path) -> str:
    return str(path.resolve().relative_to(run_dir.resolve()))


def _record_hash(record: TokenRecord) -> str:
    return canonical_hash(record.to_dict())


class EvidenceStore:
    """Persistent per-run attempt allocator and immutable evidence writer."""

    schema_version = "grpo-engine-evidence-v2"

    def __init__(self, run_dir: Path, *, stage: str, arm: str):
        self.run_dir = run_dir.resolve()
        self.root = self.run_dir / "engine-evidence"
        self.stage = str(stage)
        self.arm = str(arm)
        self.instance_id = f"engine-{uuid.uuid4().hex}"
        self.local_request_counter = 0

    def _allocate_attempt_index(self) -> int:
        attempts_root = self.root / "attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / "attempt-sequence.lock"
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            index = 1
            while True:
                candidate = attempts_root / f"attempt-{index:06d}"
                try:
                    candidate.mkdir()
                    break
                except FileExistsError:
                    index += 1
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return index

    def start_attempt(
        self,
        *,
        episode_id: str,
        task_id: str,
        sample_index: int,
        messages: list[dict[str, str]],
        generation_settings: dict[str, Any],
        model_identity: dict[str, Any],
    ) -> AttemptHandle:
        self.local_request_counter += 1
        logical_index = self.local_request_counter
        attempt_index = self._allocate_attempt_index()
        attempt_id = f"attempt-{attempt_index:06d}"
        request_id = (
            f"{_slug(self.stage)}-{_slug(self.arm)}-{_slug(episode_id)}-"
            f"a{attempt_index:06d}-{uuid.uuid4().hex}"
        )
        response_id = f"response-{uuid.uuid4().hex}"
        # Use the durable request ID itself as the filename stem.  The
        # response/event IDs remain separate fields, but no path can be
        # mistaken for a reused process-local request counter.
        stem = request_id
        request_path = self.root / "requests" / f"{stem}.json"
        response_path = self.root / "responses" / f"{stem}.json"
        engine_event_path = self.root / "events" / f"{stem}.json"
        failure_path = self.root / "failures" / f"{stem}.json"
        handle = AttemptHandle(
            stage=self.stage,
            arm=self.arm,
            attempt_index=attempt_index,
            attempt_id=attempt_id,
            request_id=request_id,
            response_id=response_id,
            logical_request_index=logical_index,
            request_path=_relative(self.run_dir, request_path),
            response_path=_relative(self.run_dir, response_path),
            engine_event_path=_relative(self.run_dir, engine_event_path),
            failure_path=_relative(self.run_dir, failure_path),
        )
        intent = {
            "schema_version": self.schema_version,
            "record_type": "INTENT",
            "status": "attempted",
            "attempted": True,
            "completed": False,
            "incomplete": True,
            "created_at_epoch": time.time(),
            "engine_instance_id": self.instance_id,
            "stage": handle.stage,
            "arm": handle.arm,
            "episode_id": episode_id,
            "task_id": task_id,
            "sample_index": int(sample_index),
            "attempt_index": handle.attempt_index,
            "attempt_id": handle.attempt_id,
            "logical_request_index": handle.logical_request_index,
            "request_id": handle.request_id,
            "response_id": handle.response_id,
            "request_path": handle.request_path,
            "response_path": handle.response_path,
            "engine_event_path": handle.engine_event_path,
            "model_identity": copy.deepcopy(model_identity),
            "generation_settings": copy.deepcopy(generation_settings),
            "generation_settings_sha256": canonical_hash(generation_settings),
            "messages": copy.deepcopy(messages),
            "messages_sha256": canonical_hash(messages),
        }
        try:
            exclusive_write_json(self.run_dir / handle.request_path, intent)
            request_sha = sha256_file(self.run_dir / handle.request_path)
            append_jsonl(
                self.root / "journal.jsonl",
                {
                    "schema_version": self.schema_version,
                    "event": "attempt_intent_persisted",
                    "attempted": True,
                    "completed": False,
                    "incomplete": True,
                    "at_epoch": time.time(),
                    "stage": handle.stage,
                    "arm": handle.arm,
                    "episode_id": episode_id,
                    "task_id": task_id,
                    "attempt_index": handle.attempt_index,
                    "attempt_id": handle.attempt_id,
                    "logical_request_index": handle.logical_request_index,
                    "request_id": handle.request_id,
                    "response_id": handle.response_id,
                    "request_path": handle.request_path,
                    "request_sha256": request_sha,
                    "response_path": handle.response_path,
                    "engine_event_path": handle.engine_event_path,
                    "messages_sha256": intent["messages_sha256"],
                    "generation_settings_sha256": intent["generation_settings_sha256"],
                    "model_identity": copy.deepcopy(model_identity),
                },
            )
            return handle
        except (ImmutableWriteError, OSError, TypeError, ValueError) as exc:
            raise EvidencePersistenceError(
                f"intent_persist_failed:{handle.request_id}:{type(exc).__name__}:{exc}"
            ) from exc

    def _common_metadata(
        self, handle: AttemptHandle, *, request_sha256: str, content_sha256: str | None
    ) -> dict[str, Any]:
        return {
            "evidence_schema_version": self.schema_version,
            "engine_instance_id": self.instance_id,
            "stage": handle.stage,
            "arm": handle.arm,
            "attempt_index": handle.attempt_index,
            "attempt_id": handle.attempt_id,
            "logical_request_index": handle.logical_request_index,
            "physical_request_index": handle.attempt_index,
            "request_path": handle.request_path,
            "request_sha256": request_sha256,
            "response_path": handle.response_path,
            "engine_event_path": handle.engine_event_path,
            "content_sha256": content_sha256,
        }

    def persist_response(
        self,
        handle: AttemptHandle,
        record: TokenRecord,
        *,
        model_id: str,
        model_identity: dict[str, Any],
        generation_settings: dict[str, Any],
        content: str | None,
        error: Any,
        model_execution_started: bool,
        model_execution_completed: bool,
        completion_status: str,
    ) -> dict[str, Any]:
        """Persist response and engine event; never replace either artifact."""

        request_path = self.run_dir / handle.request_path
        request_sha = sha256_file(request_path)
        content_sha = sha256_text(content)
        record.metadata.update(
            self._common_metadata(
                handle, request_sha256=request_sha, content_sha256=content_sha
            )
        )
        token_payload = record.to_dict()
        token_hash = canonical_hash(token_payload)
        response = {
            "schema_version": self.schema_version,
            "record_type": "RESPONSE",
            "status": completion_status,
            "attempted": True,
            "completed": True,
            "incomplete": False,
            "completed_at_epoch": time.time(),
            "stage": handle.stage,
            "arm": handle.arm,
            "episode_id": record.episode_id,
            "task_id": record.task_id,
            "sample_index": record.sample_index,
            "attempt_index": handle.attempt_index,
            "attempt_id": handle.attempt_id,
            "logical_request_index": handle.logical_request_index,
            "request_id": handle.request_id,
            "response_id": handle.response_id,
            "request_path": handle.request_path,
            "request_sha256": request_sha,
            "response_path": handle.response_path,
            "engine_event_path": handle.engine_event_path,
            "model_id": model_id,
            "model_identity": copy.deepcopy(model_identity),
            "generation_settings": copy.deepcopy(generation_settings),
            "generation_settings_sha256": canonical_hash(generation_settings),
            "content": content,
            "content_sha256": content_sha,
            "error": error,
            "model_execution_started": bool(model_execution_started),
            "model_execution_completed": bool(model_execution_completed),
            "token_record": token_payload,
            "token_record_sha256": token_hash,
        }
        try:
            response_file = self.run_dir / handle.response_path
            event_file = self.run_dir / handle.engine_event_path
            exclusive_write_json(response_file, response)
            response_sha = sha256_file(response_file)
            event = {
                "schema_version": self.schema_version,
                "record_type": "ENGINE_EVENT",
                "status": completion_status,
                "attempted": True,
                "completed": True,
                "incomplete": False,
                "at_epoch": time.time(),
                "stage": handle.stage,
                "arm": handle.arm,
                "episode_id": record.episode_id,
                "task_id": record.task_id,
                "sample_index": record.sample_index,
                "attempt_index": handle.attempt_index,
                "attempt_id": handle.attempt_id,
                "logical_request_index": handle.logical_request_index,
                "engine_instance_id": self.instance_id,
                "request_id": handle.request_id,
                "response_id": handle.response_id,
                "request_path": handle.request_path,
                "request_sha256": request_sha,
                "response_path": handle.response_path,
                "response_sha256": response_sha,
                "engine_event_path": handle.engine_event_path,
                "model_id": model_id,
                "model_identity": copy.deepcopy(model_identity),
                "generation_settings_sha256": canonical_hash(generation_settings),
                "content_sha256": content_sha,
                "token_record_sha256": token_hash,
                "model_execution_started": bool(model_execution_started),
                "model_execution_completed": bool(model_execution_completed),
            }
            exclusive_write_json(event_file, event)
            append_jsonl(
                self.root / "journal.jsonl",
                {
                    "schema_version": self.schema_version,
                    "event": "attempt_completed",
                    "attempted": True,
                    "completed": True,
                    "incomplete": False,
                    "at_epoch": time.time(),
                    **{
                        key: event[key]
                        for key in (
                            "stage",
                            "arm",
                            "episode_id",
                            "task_id",
                            "sample_index",
                            "attempt_index",
                            "attempt_id",
                            "logical_request_index",
                            "request_id",
                            "response_id",
                            "request_path",
                            "request_sha256",
                            "response_path",
                            "response_sha256",
                            "engine_event_path",
                            "content_sha256",
                            "token_record_sha256",
                            "model_execution_started",
                            "model_execution_completed",
                        )
                    },
                },
            )
            return {
                "request_path": handle.request_path,
                "request_sha256": request_sha,
                "response_path": handle.response_path,
                "response_sha256": response_sha,
                "engine_event_path": handle.engine_event_path,
                "content_sha256": content_sha,
                "token_record_sha256": token_hash,
                "completion_status": completion_status,
            }
        except (ImmutableWriteError, OSError, TypeError, ValueError) as exc:
            # Preserve the already-created request/response bytes.  The caller
            # must treat a missing event/journal as a technical unknown and
            # stop scheduling further model or verifier work.
            raise EvidencePersistenceError(
                f"response_or_event_persist_failed:{handle.request_id}:{type(exc).__name__}:{exc}"
            ) from exc

    def persist_incomplete(
        self,
        handle: AttemptHandle,
        *,
        error: Any,
        model_execution_started: bool,
        model_execution_completed: bool = False,
    ) -> None:
        """Persist an explicit incomplete-call event after interruption."""

        event = {
            "schema_version": self.schema_version,
            "record_type": "ENGINE_EVENT",
            "status": "incomplete",
            "attempted": True,
            "completed": False,
            "incomplete": True,
            "at_epoch": time.time(),
            "stage": handle.stage,
            "arm": handle.arm,
            "attempt_index": handle.attempt_index,
            "attempt_id": handle.attempt_id,
            "logical_request_index": handle.logical_request_index,
            "request_id": handle.request_id,
            "response_id": handle.response_id,
            "request_path": handle.request_path,
            "response_path": handle.response_path,
            "engine_event_path": handle.engine_event_path,
            "model_execution_started": bool(model_execution_started),
            "model_execution_completed": bool(model_execution_completed),
            "error": error,
        }
        try:
            exclusive_write_json(self.run_dir / handle.engine_event_path, event)
            append_jsonl(
                self.root / "journal.jsonl",
                {
                    "schema_version": self.schema_version,
                    "event": "attempt_incomplete",
                    "attempted": True,
                    "completed": False,
                    "incomplete": True,
                    "at_epoch": time.time(),
                    **event,
                },
            )
        except (ImmutableWriteError, OSError, TypeError, ValueError) as exc:
            raise EvidencePersistenceError(
                f"incomplete_event_persist_failed:{handle.request_id}:{type(exc).__name__}:{exc}"
            ) from exc


def _qualification_failure(message: str) -> QualificationError:
    return QualificationError(message)


def _require(value: Any, label: str) -> Any:
    if value is None or value == "":
        raise _qualification_failure(f"evidence_field_missing:{label}")
    return value


def qualify_episode_evidence(
    episode: EpisodeSample,
    *,
    run_dir: Path,
    expected_stage: str | None = None,
    expected_arm: str | None = None,
    trajectory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all model evidence before allowing a verifier call.

    Business-invalid JSON/Action/SQL remains qualified when its model call and
    trajectory are complete.  Missing, changed, or conflicting evidence is a
    technical unknown and raises ``QualificationError`` instead.
    """

    try:
        episode.validate()
    except Exception as exc:
        raise _qualification_failure(f"episode_contract_invalid:{exc}") from exc
    if not episode.token_records:
        raise _qualification_failure(f"episode_model_evidence_missing:{episode.episode_id}")
    trajectory = episode.metadata.get("trajectory")
    if not isinstance(trajectory, dict):
        raise _qualification_failure(f"trajectory_missing:{episode.episode_id}")
    if trajectory_path is not None:
        path = safe_relative_path(run_dir.resolve(), trajectory_path)
        value = read_json(path)
        if not isinstance(value, dict):
            raise _qualification_failure(f"trajectory_file_invalid:{trajectory_path}")
        if canonical_hash(value) != canonical_hash(trajectory):
            raise _qualification_failure(f"trajectory_hash_mismatch:{trajectory_path}")
        trajectory_sha = sha256_file(path)
    else:
        trajectory_sha = canonical_hash(trajectory)

    attempts_value = trajectory.get("model_attempts")
    if not isinstance(attempts_value, list):
        raise _qualification_failure("trajectory_model_attempts_missing")
    attempts_by_request: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts_value:
        if not isinstance(attempt, dict):
            raise _qualification_failure("trajectory_model_attempt_not_object")
        request_id = attempt.get("request_id")
        if isinstance(request_id, str) and request_id:
            attempts_by_request.setdefault(request_id, []).append(attempt)

    qualified: list[dict[str, Any]] = []
    for record in episode.token_records:
        metadata = record.metadata
        request_id = _require(metadata.get("request_id", record.request_id), "request_id")
        response_id = _require(metadata.get("response_id", record.response_id), "response_id")
        stage = _require(metadata.get("stage"), "stage")
        arm = _require(metadata.get("arm"), "arm")
        if expected_stage is not None and stage != expected_stage:
            raise _qualification_failure(f"stage_mismatch:{record.episode_id}:{stage}")
        if expected_arm is not None and arm != expected_arm:
            raise _qualification_failure(f"arm_mismatch:{record.episode_id}:{arm}")
        request_path = _require(metadata.get("request_path"), "request_path")
        response_path = _require(metadata.get("response_path"), "response_path")
        event_path = _require(metadata.get("engine_event_path"), "engine_event_path")
        request_file = safe_relative_path(run_dir.resolve(), request_path)
        response_file = safe_relative_path(run_dir.resolve(), response_path)
        event_file = safe_relative_path(run_dir.resolve(), event_path)
        request = read_json(request_file)
        response = read_json(response_file)
        event = read_json(event_file)
        if not all(isinstance(value, dict) for value in (request, response, event)):
            raise _qualification_failure(f"evidence_file_missing:{request_id}")
        if request.get("record_type") != "INTENT":
            raise _qualification_failure(f"intent_type_mismatch:{request_id}")
        if response.get("record_type") != "RESPONSE":
            raise _qualification_failure(f"response_type_mismatch:{request_id}")
        if event.get("record_type") != "ENGINE_EVENT":
            raise _qualification_failure(f"engine_event_type_mismatch:{request_id}")
        if event.get("completed") is not True or event.get("incomplete") is True:
            raise _qualification_failure(f"incomplete_model_attempt:{request_id}")
        expected_ids = {
            "request_id": request_id,
            "response_id": response_id,
            "episode_id": record.episode_id,
            "task_id": record.task_id,
            "stage": stage,
            "arm": arm,
        }
        for name, expected in expected_ids.items():
            if any(value.get(name) != expected for value in (request, response, event)):
                raise _qualification_failure(f"evidence_identity_mismatch:{name}:{request_id}")
        if request.get("messages_sha256") != canonical_hash(request.get("messages")):
            raise _qualification_failure(f"messages_hash_mismatch:{request_id}")
        if request.get("generation_settings_sha256") != canonical_hash(
            request.get("generation_settings")
        ):
            raise _qualification_failure(f"generation_settings_hash_mismatch:{request_id}")
        request_sha = sha256_file(request_file)
        response_sha = sha256_file(response_file)
        if event.get("request_sha256") != request_sha:
            raise _qualification_failure(f"request_file_hash_mismatch:{request_id}")
        if event.get("response_sha256") != response_sha:
            raise _qualification_failure(f"response_file_hash_mismatch:{request_id}")
        if metadata.get("request_sha256") != request_sha:
            raise _qualification_failure(f"token_request_hash_mismatch:{request_id}")
        if response.get("content_sha256") != sha256_text(response.get("content")):
            raise _qualification_failure(f"response_content_hash_mismatch:{request_id}")
        if event.get("content_sha256") != response.get("content_sha256"):
            raise _qualification_failure(f"event_content_hash_mismatch:{request_id}")
        if metadata.get("content_sha256") != response.get("content_sha256"):
            raise _qualification_failure(f"token_content_hash_mismatch:{request_id}")
        token_payload = record.to_dict()
        token_hash = canonical_hash(token_payload)
        if response.get("token_record_sha256") != token_hash:
            raise _qualification_failure(f"token_record_hash_mismatch:{request_id}")
        if event.get("token_record_sha256") != token_hash:
            raise _qualification_failure(f"event_token_hash_mismatch:{request_id}")
        if canonical_hash(response.get("token_record")) != token_hash:
            raise _qualification_failure(f"response_token_binding_mismatch:{request_id}")
        if response.get("content") != record.action_text:
            raise _qualification_failure(f"action_content_mismatch:{request_id}")
        matches = attempts_by_request.get(request_id, [])
        if len(matches) != 1:
            raise _qualification_failure(
                f"trajectory_attempt_cardinality:{request_id}:{len(matches)}"
            )
        action_attempt = matches[0]
        if action_attempt.get("response_id") != response_id:
            raise _qualification_failure(f"trajectory_response_binding_mismatch:{request_id}")
        if action_attempt.get("raw_output") != record.action_text:
            raise _qualification_failure(f"trajectory_content_mismatch:{request_id}")
        qualified.append(
            {
                "request_id": request_id,
                "response_id": response_id,
                "attempt_index": metadata.get("attempt_index"),
                "request_path": request_path,
                "response_path": response_path,
                "engine_event_path": event_path,
                "request_sha256": request_sha,
                "response_sha256": response_sha,
                "content_sha256": response.get("content_sha256"),
                "token_record_sha256": token_hash,
                "completion_status": response.get("status"),
            }
        )

    all_request_ids = set(attempts_by_request)
    qualified_ids = {row["request_id"] for row in qualified}
    if all_request_ids != qualified_ids:
        raise _qualification_failure("trajectory_has_unbound_model_attempt")
    return {
        "status": "QUALIFIED",
        "episode_id": episode.episode_id,
        "task_id": episode.task_id,
        "stage": expected_stage,
        "arm": expected_arm,
        "attempted_model_calls": len(qualified),
        "completed_model_calls": sum(
            row["completion_status"] in {"completed", "completed_with_error", "not_started"}
            for row in qualified
        ),
        "incomplete_model_calls": 0,
        "trajectory_sha256": trajectory_sha,
        "model_evidence": qualified,
    }


def evidence_journal_summary(run_dir: Path) -> dict[str, Any]:
    """Summarize attempted/completed/incomplete calls without discovering files."""

    journal = run_dir / "engine-evidence" / "journal.jsonl"
    attempted = completed = incomplete = 0
    rows: list[dict[str, Any]] = []
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = read_json_line(line)
            rows.append(value)
            if value.get("event") == "attempt_intent_persisted":
                attempted += 1
            elif value.get("event") == "attempt_completed":
                completed += 1
            elif value.get("event") == "attempt_incomplete":
                incomplete += 1
    return {
        "journal_path": str(journal),
        "attempted": attempted,
        "completed": completed,
        "incomplete": incomplete,
        "journal_rows": len(rows),
    }


def read_json_line(line: str) -> dict[str, Any]:
    import json

    value = json.loads(line)
    if not isinstance(value, dict):
        raise EvidenceError("evidence_journal_row_not_object")
    return value
