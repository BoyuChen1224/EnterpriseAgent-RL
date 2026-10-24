"""Download an immutable safetensors snapshot without loading model weights."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from enterprise_agent_rl.model_server import ModelSettings


def cache_file_manifest(snapshot: Path) -> dict:
    files = {}
    for file in sorted(snapshot.iterdir()):
        if file.is_file():
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[file.name] = {"sha256": digest.hexdigest(), "bytes": file.stat().st_size}
    return files


def unique_cache_size(root: Path) -> int:
    """Count physical bytes once; HF snapshots use symlinks to shared blobs."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for file in root.rglob("*"):
        if not file.is_file():
            continue
        stat = file.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            seen.add(identity)
            total += stat.st_size
    return total


def repository_dir(cache: Path, model_id: str) -> Path:
    return cache / ("models--" + model_id.replace("/", "--"))


def build_manifest(cache: Path, config: Path) -> dict:
    cfg = ModelSettings.load(config)
    snapshot = Path(
        snapshot_download(
            cfg.model_id,
            revision=cfg.revision,
            cache_dir=cache,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "LICENSE"],
            resume_download=True,
        )
    )
    files = cache_file_manifest(snapshot)
    weight_files = sorted(name for name in files if name.endswith(".safetensors"))
    if not weight_files:
        raise SystemExit("Incomplete pinned model snapshot: no safetensors weights")
    tokenizer_files = sorted(
        name
        for name in files
        if name.startswith(("tokenizer", "special_tokens"))
        or name in {"chat_template.jinja", "merges.txt", "vocab.json"}
    )
    manifest = {
        "schema_version": "3.0",
        "model": cfg.model_id,
        "revision": cfg.revision,
        "snapshot": str(snapshot),
        "snapshot_name": snapshot.name,
        "weight_files": weight_files,
        "tokenizer_files": tokenizer_files,
        "files": files,
        "cache_size_bytes_unique": unique_cache_size(repository_dir(cache, cfg.model_id)),
        "integrity": "sha256-per-snapshot-file",
        "trust_remote_code": False,
        "safetensors_only": True,
        "enable_thinking": cfg.generation.enable_thinking,
    }
    manifest_path = cache / "qwen-cache-manifest.json"
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("artifacts/model-cache-qwen3-8b"))
    parser.add_argument("--config", type=Path, default=Path("configs/model/qwen3-8b.toml"))
    args = parser.parse_args()
    manifest = build_manifest(args.cache, args.config)
    print(manifest["snapshot"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
