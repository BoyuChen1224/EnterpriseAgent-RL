"""Verify a completed, pinned Hugging Face snapshot without downloading or loading it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from enterprise_agent_rl.model_server import ModelSettings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_size(root: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            seen.add(identity)
            total += stat.st_size
    return total


def repository_dir(cache: Path, model_id: str) -> Path:
    return cache / ("models--" + model_id.replace("/", "--"))


def verify(cache: Path, config: Path) -> dict:
    cfg = ModelSettings.load(config)
    repo = repository_dir(cache, cfg.model_id)
    snapshot = repo / "snapshots" / cfg.revision
    required = {
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }
    missing = sorted(name for name in required if not (snapshot / name).is_file())
    weight_files = sorted(path.name for path in snapshot.glob("*.safetensors") if path.is_file())
    incomplete_blobs = sorted(path.name for path in (repo / "blobs").glob("*.incomplete"))
    file_records = {}
    for path in sorted(snapshot.iterdir()) if snapshot.is_dir() else []:
        if path.is_file():
            file_records[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "target": os.path.realpath(path),
            }
    manifest_path = cache / "qwen-cache-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    manifest_matches = bool(
        manifest
        and manifest.get("model") == cfg.model_id
        and manifest.get("revision") == cfg.revision
        and all(
            name in manifest.get("files", {})
            and manifest["files"][name]["sha256"] == record["sha256"]
            and manifest["files"][name]["bytes"] == record["bytes"]
            for name, record in file_records.items()
        )
    )
    tokenizer_config = {}
    if (snapshot / "tokenizer_config.json").is_file():
        try:
            tokenizer_config = json.loads((snapshot / "tokenizer_config.json").read_text())
        except json.JSONDecodeError:
            tokenizer_config = {}
    complete = (
        snapshot.is_dir()
        and not missing
        and bool(weight_files)
        and not incomplete_blobs
        and bool(file_records)
        and manifest_matches
        and bool(tokenizer_config.get("chat_template"))
    )
    return {
        "schema_version": "1.0",
        "status": "complete" if complete else "incomplete",
        "model": cfg.model_id,
        "revision": cfg.revision,
        "dtype": cfg.dtype,
        "attention_backend": cfg.attention_backend,
        "cache": str(cache),
        "snapshot": str(snapshot),
        "required_files_missing": missing,
        "weight_files": weight_files,
        "incomplete_blobs": incomplete_blobs,
        "files": file_records,
        "manifest_path": str(manifest_path),
        "manifest_matches": manifest_matches,
        "chat_template_present": bool(tokenizer_config.get("chat_template")),
        "trust_remote_code": False,
        "safetensors_only": True,
        "cache_size_bytes_unique": unique_size(repo) if repo.is_dir() else 0,
        "integrity": "sha256-per-snapshot-file",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("artifacts/model-cache-qwen3-8b"))
    parser.add_argument("--config", type=Path, default=Path("configs/model/qwen3-8b.toml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qwen3-base-lora-pilot/t031-v1/cache-verification.json"),
    )
    args = parser.parse_args()
    report = verify(args.cache, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
