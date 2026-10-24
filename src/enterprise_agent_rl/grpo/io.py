"""Small durable I/O primitives used by the GRPO entrypoint."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class ImmutableWriteError(RuntimeError):
    """A durable evidence path already exists and must not be replaced."""


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON with fsync followed by an atomic same-directory replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def exclusive_write_json(path: Path, value: Any) -> None:
    """Create a JSON file once, without ever replacing an existing file.

    Training/evaluation manifests may be updated with ``atomic_write_json``.
    Raw model evidence is different: a reused filename is a provenance error,
    even when a caller believes the replacement would be identical.  O_EXCL
    makes that distinction visible to the caller and preserves the original
    bytes for later audit.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ImmutableWriteError(f"evidence_path_exists:{path}") from exc
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # The path remains present if the filesystem failed after creation.
        # Do not unlink it: a partial artifact is evidence of the failed write.
        raise


def immutable_write_json(path: Path, value: Any) -> None:
    """Create JSON once, allowing an exact idempotent replay only.

    This is used for deterministic trajectory paths during a qualified
    resume.  A different payload at the same path is still a hard conflict.
    Per-attempt request/response/event files use ``exclusive_write_json``
    instead, so even identical duplicate attempts are rejected.
    """

    if path.exists():
        try:
            existing = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise ImmutableWriteError(f"existing_json_unreadable:{path}") from exc
        if canonical_hash(existing) == canonical_hash(value):
            return
        raise ImmutableWriteError(f"immutable_json_conflict:{path}")
    try:
        exclusive_write_json(path, value)
    except ImmutableWriteError:
        # A concurrent writer may have won the race.  Re-read and only accept
        # the exact same immutable payload.
        existing = read_json(path)
        if canonical_hash(existing) != canonical_hash(value):
            raise


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid_jsonl:{path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"jsonl_row_not_object:{path}:{line_number}")
        rows.append(value)
    return rows


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return sha256_bytes(value.encode("utf-8"))


def fingerprint_rows(rows: Iterable[dict[str, Any]]) -> str:
    return canonical_hash(list(rows))


def safe_relative_path(root: Path, value: str | Path) -> Path:
    """Resolve an evidence path and reject absolute paths and traversal."""

    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"absolute_evidence_path:{candidate}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"evidence_path_escape:{value}") from exc
    return resolved
