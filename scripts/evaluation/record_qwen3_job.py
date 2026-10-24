#!/usr/bin/env python3
"""Record the immutable Slurm job chain for T-031/E-028."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--elapsed-s", type=float)
    parser.add_argument("--partition")
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument("--code-commit")
    args = parser.parse_args()

    path = args.run / "job-chain.json"
    chain = json.loads(path.read_text()) if path.is_file() else {
        "schema_version": "1.0",
        "experiment": "T-031/E-028",
        "run": str(args.run.resolve()),
        "run_commit": _git("rev-parse", "HEAD"),
        "run_dirty_at_record": bool(_git("status", "--porcelain")),
        "records": [],
    }
    record = next(
        (item for item in chain["records"] if item.get("job_id") == args.job_id), None
    )
    if record is None:
        record = {
            "stage": args.stage,
            "job_id": args.job_id,
            "command": args.command,
            "submitted_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        chain["records"].append(record)
    record.update(
        {
            "stage": args.stage,
            "command": args.command,
            "state": args.state,
            "updated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
    )
    if args.exit_code is not None:
        record["exit_code"] = args.exit_code
    if args.elapsed_s is not None:
        record["elapsed_s"] = args.elapsed_s
    if args.partition is not None:
        record["partition"] = args.partition
    if args.gpu_count is not None:
        record["requested_gpu_count"] = args.gpu_count
    if args.code_commit is not None:
        record["code_commit"] = args.code_commit
    chain["last_state"] = args.state
    _write(path, chain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
