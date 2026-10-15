"""Download the immutable safetensors snapshot without importing torch or loading weights."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from enterprise_agent_rl.model_server import ModelSettings

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
    )
)
files = {}
for file in sorted(path.iterdir()):
    if file.is_file():
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[file.name] = {"sha256": digest.hexdigest(), "bytes": file.stat().st_size}
if "model.safetensors" not in files:
    raise SystemExit("Incomplete pinned model snapshot")
(args.cache / "qwen-cache-manifest.json").write_text(
    json.dumps({"model": cfg.model_id, "revision": cfg.revision, "files": files}, indent=2) + "\n"
)
print(path)
