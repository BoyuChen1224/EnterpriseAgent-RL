#!/usr/bin/env python3
"""Build and audit a train-only verified SFT dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from enterprise_agent_rl.sft_data import DATASET_VERSION, build_sft_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/m2a/manifest.json"))
    parser.add_argument("--database-root", type=Path, default=Path("artifacts/m2a-databases-final"))
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=Path("artifacts/m2b-train-audit/trajectories"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"artifacts/sft-data-{DATASET_VERSION}"),
    )
    parser.add_argument(
        "--trajectory-source",
        default="deterministic_fixture_expert",
        choices=["deterministic_fixture_expert", "local_llm"],
    )
    args = parser.parse_args()
    metadata = build_sft_dataset(
        args.manifest,
        args.database_root,
        args.trajectories,
        args.output,
        expected_trajectory_source=args.trajectory_source,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
