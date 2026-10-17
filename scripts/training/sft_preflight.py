#!/usr/bin/env python3
"""CPU preflight for the frozen train-only SFT dataset.

This command loads the tokenizer only.  It never loads model weights and never
opens validation/test databases.  It is intended to run before the single
GPU training job and produces the token/mask evidence copied into the pilot
artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from enterprise_agent_rl.sft_training import (
    audit_tokenized_dataset,
    load_dataset_rows,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/sft-data-train-teacher-v2/dataset.jsonl"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/sft-data-train-teacher-v2/dataset-metadata.json"),
    )
    parser.add_argument("--cache", type=Path, default=Path("artifacts/model-cache"))
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--revision",
        default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sft-data-train-teacher-v2/sft-preflight.json"),
    )
    args = parser.parse_args()

    rows, metadata = load_dataset_rows(args.dataset, args.metadata)

    # Importing Transformers is deliberately delayed until after all JSON
    # leakage/provenance gates have passed.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=str(args.cache),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    report = audit_tokenized_dataset(
        tokenizer,
        rows,
        max_context_tokens=args.max_context_tokens,
    )
    report.update(
        {
            "dataset_path": str(args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "metadata_path": str(args.metadata),
            "model_id": args.model_id,
            "model_revision": args.revision,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_revision": args.revision,
            "tokenizer_chat_template_sha256": __import__("hashlib")
            .sha256((tokenizer.chat_template or "").encode("utf-8"))
            .hexdigest(),
            "source_split": metadata["source_split"],
            "validation_used_for_training": metadata["validation_used_for_training"],
            "test_used_for_training": metadata["test_used_for_training"],
            "teacher_source_distribution": metadata["trajectory_source_distribution"],
            "cpu_preflight": {
                "python": sys.version,
                "platform": platform.platform(),
                "hostname": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "model_weights_loaded": False,
                "gpu_used": False,
            },
            "decision": "READY_FOR_SFT_TRAINING",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
