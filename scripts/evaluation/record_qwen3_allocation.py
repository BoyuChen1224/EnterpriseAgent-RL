#!/usr/bin/env python3
"""Append one immutable resource record to the T-031 allocation ledger."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--elapsed-s", type=float, required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    args = parser.parse_args()
    path = args.run / "allocation-ledger.json"
    existing = (
        json.loads(path.read_text())
        if path.is_file()
        else {
            "schema_version": "1.0",
            "experiment": "T-031/E-028",
            "records": [],
        }
    )
    record = {
        "stage": args.stage,
        "status": args.status,
        "exit_code": args.exit_code,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "account": os.environ.get("SLURM_JOB_ACCOUNT") or os.environ.get("SLURM_ACCOUNT"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "qos": os.environ.get("SLURM_JOB_QOS") or os.environ.get("SLURM_QOS"),
        "node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count": args.gpu_count,
        "elapsed_s": args.elapsed_s,
        "gpu_seconds": args.elapsed_s * args.gpu_count,
        "gpu_minutes": args.elapsed_s * args.gpu_count / 60.0,
        "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    existing.setdefault("records", []).append(record)
    existing["total_gpu_seconds"] = sum(item.get("gpu_seconds", 0) for item in existing["records"])
    existing["total_gpu_minutes"] = existing["total_gpu_seconds"] / 60.0
    existing["budget_gpu_minutes"] = 240
    existing["within_budget"] = existing["total_gpu_seconds"] <= 14400
    args.run.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
