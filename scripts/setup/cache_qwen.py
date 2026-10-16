"""Download the immutable safetensors snapshot without importing torch or loading weights."""

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


parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=Path, default=Path("artifacts/model-cache"))
parser.add_argument("--config", type=Path, default=Path("configs/model/qwen2.5-1.5b.toml"))
args = parser.parse_args()
cfg = ModelSettings.load(args.config)
path = Path(
    snapshot_download(
        cfg.model_id,
        revision=cfg.revision,
        cache_dir=args.cache,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE"],
        resume_download=True,
    )
)
files = cache_file_manifest(path)
weight_files = [name for name in files if name.endswith(".safetensors")]
if not weight_files:
    raise SystemExit("Incomplete pinned model snapshot")
manifest = {
    "schema_version": "2.0",
    "model": cfg.model_id,
    "revision": cfg.revision,
    "snapshot": str(path),
    "weight_files": weight_files,
    "files": files,
    "cache_size_bytes_unique": unique_cache_size(
        args.cache / "models--Qwen--Qwen2.5-1.5B-Instruct"
    ),
    "integrity": "sha256-per-snapshot-file",
}
manifest_path = args.cache / "qwen-cache-manifest.json"
temporary = manifest_path.with_suffix(".tmp")
temporary.write_text(json.dumps(manifest, indent=2) + "\n")
temporary.replace(manifest_path)
print(path)
