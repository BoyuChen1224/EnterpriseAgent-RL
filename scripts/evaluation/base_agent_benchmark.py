#!/usr/bin/env python3
"""CPU-side planning/dry-run entry point for the formal Base Agent benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from enterprise_agent_rl.base_benchmark import (
    EXPECTED_SPLIT_COUNTS,
    prepare_validation_plan,
    write_incomplete_report,
)
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.model_server import ModelSettings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/m2a/manifest.json"))
    parser.add_argument("--database-root", type=Path, default=Path("artifacts/m2a-databases"))
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen2.5-1.5b.toml")
    )
    parser.add_argument("--agent-config", type=Path, default=Path("configs/agent/qwen.toml"))
    parser.add_argument("--split", choices=["validation", "train", "test"], default="validation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize-incomplete", action="store_true")
    parser.add_argument("--error-type", default="supervisor_failure")
    parser.add_argument("--error-message", default="benchmark supervisor exited before completion")
    args = parser.parse_args()
    if args.split != "validation":
        raise SystemExit("base benchmark is validation-only; refusing non-validation split")
    if args.finalize_incomplete:
        if args.output is None:
            raise SystemExit("--output is required with --finalize-incomplete")
        model_config = ModelSettings.load(args.model_config)
        settings = Settings.load(args.agent_config)
        report = write_incomplete_report(
            args.output,
            args.manifest,
            args.database_root,
            model_config,
            settings,
            {"type": args.error_type, "message": args.error_message},
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if not args.dry_run:
        raise SystemExit(
            "GPU execution is orchestrated by scripts/inference/run_qwen.py base-benchmark"
        )

    model_config = ModelSettings.load(args.model_config)
    settings = Settings.load(args.agent_config)
    plan = prepare_validation_plan(args.manifest, args.database_root)
    result = {
        "dry_run": True,
        "benchmark": "Base Agent baseline",
        "split": "validation",
        "validation_task_count": len(plan["records"]),
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "validation_task_ids": plan["task_ids"],
        "validation_database_ids": plan["database_ids"],
        "test_split_accessed": False,
        "train_split_accessed": False,
        "policy_kind": "local_llm",
        "deterministic_fixture_expert": False,
        "model_config": model_config.model_dump(),
        "agent_settings": settings.model_dump(),
        "artifact_root": str(args.output) if args.output else None,
    }
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "dry-run.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
