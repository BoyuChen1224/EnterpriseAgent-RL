#!/usr/bin/env python3
"""Append immutable T-032 Slurm allocation evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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
    parser.add_argument("--slurm-evidence", type=Path)
    args = parser.parse_args()
    ledger_path = args.run / "allocation-ledger.json"
    existing = (
        json.loads(ledger_path.read_text())
        if ledger_path.is_file()
        else {
            "schema_version": "1.0",
            "experiment": "T-032/E-029",
            "records": [],
        }
    )
    evidence_sha = None
    if args.slurm_evidence and args.slurm_evidence.is_file():
        evidence_sha = hashlib.sha256(args.slurm_evidence.read_bytes()).hexdigest()
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
        "wrapper_elapsed_s": args.elapsed_s,
        "wrapper_gpu_seconds": args.elapsed_s * args.gpu_count,
        "wrapper_gpu_minutes": args.elapsed_s * args.gpu_count / 60.0,
        "slurm_evidence_path": str(args.slurm_evidence.resolve()) if args.slurm_evidence else None,
        "slurm_evidence_sha256": evidence_sha,
        "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    existing.setdefault("records", []).append(record)
    existing["wrapper_gpu_seconds"] = sum(
        item.get("wrapper_gpu_seconds", 0) for item in existing["records"]
    )
    existing["wrapper_gpu_minutes"] = existing["wrapper_gpu_seconds"] / 60.0
    existing["budget_gpu_minutes"] = 240
    existing["within_budget_by_wrapper"] = existing["wrapper_gpu_seconds"] <= 14400
    args.run.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(ledger_path)

    chain_path = args.run / "job-chain.json"
    chain = (
        json.loads(chain_path.read_text())
        if chain_path.is_file()
        else {
            "schema_version": "1.0",
            "experiment": "T-032/E-029",
            "records": [],
        }
    )
    chain["records"].append(
        {
            "stage": args.stage,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "status": args.status,
            "exit_code": args.exit_code,
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "qos": os.environ.get("SLURM_JOB_QOS") or os.environ.get("SLURM_QOS"),
            "gpu_count": args.gpu_count,
            "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
    )
    chain["latest_job_id"] = os.environ.get("SLURM_JOB_ID")
    chain["job_count"] = len(chain["records"])
    chain_path.write_text(json.dumps(chain, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
