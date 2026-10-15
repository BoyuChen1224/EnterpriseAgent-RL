"""Materialize deterministic DBs; verify against frozen labels, never silently refreeze."""

import argparse
import json
from pathlib import Path

from enterprise_agent_rl.benchmark import build_manifest

parser = argparse.ArgumentParser()
parser.add_argument("--database-root", type=Path, default=Path("artifacts/m2a-databases"))
parser.add_argument("--manifest", type=Path, default=Path("benchmarks/m2a/manifest.json"))
args = parser.parse_args()
manifest = build_manifest(args.database_root)
if args.manifest.exists():
    frozen = json.loads(args.manifest.read_text())
    if manifest != frozen:
        raise SystemExit("Frozen benchmark mismatch; do not overwrite the test set")
else:
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    args.manifest.with_suffix(".sha256").write_text(manifest["manifest_sha256"] + "\n")
print(
    json.dumps(
        {
            "tasks": manifest["split_counts"],
            "sha256": manifest["manifest_sha256"],
            "validation": "all SQL labels matched independent Python oracle",
        }
    )
)
