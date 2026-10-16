"""Write a reproducible, no-GPU experiment freeze for the Base validation stage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.model_server import ModelSettings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def installed_versions(names: list[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen2.5-1.5b.toml")
    )
    parser.add_argument("--agent-config", type=Path, default=Path("configs/agent/qwen.toml"))
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/m2a/manifest.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/m2b-preparation/experiment-freeze.json")
    )
    args = parser.parse_args()
    model = ModelSettings.load(args.model_config)
    agent = Settings.load(args.agent_config)
    manifest_sha = args.manifest.with_suffix(".sha256").read_text().strip()
    report = {
        "schema_version": "1.0",
        "frozen_at": datetime.now(UTC).isoformat(),
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
        "python": sys.version,
        "model": {
            "id": model.model_id,
            "revision": model.revision,
            "dtype": model.dtype,
            "attention": model.attention,
            "config_sha256": sha256(args.model_config),
        },
        "generation": model.generation.model_dump(),
        "agent": agent.model_dump(),
        "prompt_contract": {
            "system_prompt_source": "src/enterprise_agent_rl/agent.py:SYSTEM_PROMPT",
            "protocol_source": "src/enterprise_agent_rl/model_protocol.py",
            "local_files_only": True,
        },
        "benchmark": {
            "manifest": str(args.manifest),
            "manifest_sha256": manifest_sha,
            "validation_split": 20,
            "test_split": 20,
            "test_policy": "not_started; freeze before formal test",
        },
        "dependencies": {
            "cpu_lock_sha256": sha256(Path("requirements/cpu-py312-linux.lock")),
            "gpu_lock_sha256": sha256(Path("requirements/gpu-py312-linux.lock")),
            "installed_versions": installed_versions(
                [
                    "torch",
                    "transformers",
                    "tokenizers",
                    "safetensors",
                    "numpy",
                    "pydantic",
                    "fastapi",
                    "uvicorn",
                ]
            ),
        },
        "gpu_authorization": {
            "status": "not_granted",
            "submitted_jobs": [],
            "gpu_tests": "pending",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
