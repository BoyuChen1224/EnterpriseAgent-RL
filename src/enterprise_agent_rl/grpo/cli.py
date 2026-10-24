"""Command-line surface for validation, online training, and offline closeout."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .aggregation import aggregate_source, replay_t021, three_arm_comparison
from .config import GrpoConfig, RunContext, load_config
from .contracts import TaskInput
from .io import (
    append_jsonl,
    atomic_write_json,
    canonical_hash,
    read_json,
    read_jsonl,
    safe_relative_path,
    sha256_file,
    sha256_text,
)
from .orchestration import run_frozen_evaluation, run_online_training
from .saved_scoring import score_saved_trajectory
from .task_io import load_task_inputs


def _context(args: argparse.Namespace, config_path: Path) -> RunContext:
    repo_root = Path(args.repo_root).expanduser().resolve()
    run_dir = Path(args.run).expanduser()
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    return RunContext(repo_root=repo_root, run_dir=run_dir, config_path=config_path)


def _config(args: argparse.Namespace) -> tuple[GrpoConfig, RunContext]:
    config_path = Path(args.config).expanduser().resolve()
    return load_config(config_path, _context(args, config_path))


def _synthetic_tasks() -> list[TaskInput]:
    return [
        TaskInput(
            task_id=f"cpu-task-{index}",
            group_id=f"g{index}",
            public_task={
                "task_id": f"cpu-task-{index}",
                "prompt": "synthetic CPU integration task",
            },
        )
        for index in range(4)
    ]


def _cpu_components(guard, context, _config):
    from .testing import FakeRolloutEngine, FakeWorker, tiny_qwen_policy

    policy = tiny_qwen_policy(context.run_dir, guard, _config)
    return policy, FakeRolloutEngine(), FakeWorker()


def _cpu_evaluation_factory(arm, _guard, _context, _config):
    from .testing import FakeRolloutEngine, FakeWorker

    return {
        "policy": None,
        "engine": FakeRolloutEngine(model_ref_id=f"cpu-evaluation-{arm}"),
        "worker": FakeWorker(),
        "metadata": {
            "evaluation_arm": arm,
            "model_ref_id": f"cpu-evaluation-{arm}",
            "model_object_id": f"cpu-object-{arm}",
            "model_weight_fingerprint": canonical_hash({"arm": arm, "fixture": True}),
            "parent_merge_count": 1,
            "delta_merge_count": int(arm == "RL"),
            "fresh_base": True,
            "source_kind": "cpu_fake",
        },
    }


def _cpu_frozen_evaluation_factory(arm, _guard, context, _config):
    from .testing import EvidenceFakeRolloutEngine, FakeWorker

    merge_parent = int(arm in {"N", "RL"})
    merge_delta = int(arm == "RL")
    model_ref_id = f"cpu-frozen-{arm}"
    return {
        "policy": None,
        "engine": EvidenceFakeRolloutEngine(
            context.run_dir, arm=arm, model_ref_id=model_ref_id
        ),
        "worker": FakeWorker(),
        "metadata": {
            "evaluation_arm": arm,
            "model_ref_id": model_ref_id,
            "model_object_id": f"cpu-object-{arm}",
            "model_weight_fingerprint": canonical_hash({"arm": arm, "fixture": True}),
            "parent_merge_count": merge_parent,
            "delta_merge_count": merge_delta,
            "fresh_base": True,
            "source_kind": "cpu_fake",
            "training_calls": 0,
        },
    }


def _verifier(config):
    def verifier(task, trajectory):
        from enterprise_agent_rl.common import Settings
        from enterprise_agent_rl.data import BenchmarkTask
        from enterprise_agent_rl.rewards import verify

        if not task.verifier_payload or not task.database_path:
            raise RuntimeError("verifier_payload_and_database_required")
        benchmark = BenchmarkTask.model_validate(task.verifier_payload)
        return verify(
            benchmark, trajectory, Path(task.database_path), Settings.model_validate(config.runtime)
        )

    return verifier


def _real_components(guard, context, config):
    from .torch_backend import HfRolloutEngine, QwenPolicy, RealVerifierWorker

    policy = QwenPolicy.load(context, config, guard)
    engine = HfRolloutEngine(policy=policy, run_dir=context.run_dir, config=config)
    return policy, engine, RealVerifierWorker(_verifier(config), guard=guard)


def _real_evaluation_factory(arm, guard, context, config):
    from .torch_backend import HfRolloutEngine, QwenPolicy, RealVerifierWorker

    policy = QwenPolicy.load_evaluation(context, config, guard, arm=arm)
    metadata = policy.evaluation_identity(arm)
    metadata.update(
        {
            "generation_settings": config.evaluation.get("generation", {}),
            "evaluation_mode": "greedy",
            "evaluation_seed": int(config.evaluation.get("generation", {}).get("seed", 17)),
            "source_kind": "real",
        }
    )
    return {
        "policy": policy,
        "engine": HfRolloutEngine(
            policy=policy, run_dir=context.run_dir, config=config, evaluation=True
        ),
        "worker": RealVerifierWorker(_verifier(config), guard=guard),
        "metadata": metadata,
    }


def _optional_config_path(config: GrpoConfig, context: RunContext, key: str) -> Path | None:
    value = config.data.get(key) or config.evaluation.get(key)
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else context.repo_root / path


def _copy_databases(
    tasks: list[TaskInput], run_dir: Path
) -> tuple[list[TaskInput], list[dict[str, str]]]:
    target_root = run_dir / "inputs" / "databases"
    copied: dict[str, Path] = {}
    evidence: list[dict[str, str]] = []
    result: list[TaskInput] = []
    for task in tasks:
        if not task.database_path:
            result.append(task)
            continue
        source = Path(task.database_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = copied.get(str(source))
        if target is None:
            target = target_root / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            copied[str(source)] = target
            source_sha = sha256_file(source)
            target_sha = sha256_file(target)
            if source_sha != target_sha:
                raise RuntimeError(f"database_copy_hash_mismatch:{source}")
            evidence.append(
                {
                    "source": str(source),
                    "copy": str(target),
                    "source_sha256": source_sha,
                    "copy_sha256": target_sha,
                }
            )
        result.append(dataclasses.replace(task, database_path=str(target)))
    return result, evidence


def _write_input_manifest(
    context: RunContext,
    config: GrpoConfig,
    train_tasks: list[TaskInput],
    evaluation_tasks: list[TaskInput],
    database_evidence: list[dict[str, str]],
) -> None:
    rows = []
    for arm, tasks in (("train", train_tasks), ("evaluation", evaluation_tasks)):
        for task in tasks:
            rows.append(
                {
                    "arm": arm,
                    "task_id": task.task_id,
                    "group_id": task.group_id,
                    "public_task_sha256": canonical_hash(task.public_task),
                    "database_path": task.database_path,
                    "database_sha256": sha256_file(Path(task.database_path))
                    if task.database_path
                    else None,
                    "source_metadata": {
                        key: value
                        for key, value in task.metadata.items()
                        if key
                        in {
                            "database_id",
                            "database_sha256",
                            "data_fingerprint",
                            "database_version",
                            "source",
                            "label_validation",
                        }
                    },
                }
            )
    atomic_write_json(
        context.run_dir / "inputs" / "fingerprint-index.json",
        {
            "schema_version": "t023-input-fingerprint-v1",
            "train_task_ids": [task.task_id for task in train_tasks],
            "evaluation_task_ids": [task.task_id for task in evaluation_tasks],
            "source_train_index": str(config.path("train_index", context)),
            "source_evaluation_plan": str(config.path("evaluation_plan", context)),
            "source_public_panel": str(config.path("public_panel", context)),
            "rows": rows,
            "database_copies": database_evidence,
            "hidden_reference_boundary": (
                "TaskInput.verifier_payload is never passed to policy/engine."
            ),
        },
    )


def _copy_frozen_source(source: Path, target: Path) -> dict[str, str]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha = sha256_file(source)
    if target.exists():
        target_sha = sha256_file(target)
        if target_sha != source_sha:
            raise RuntimeError(f"frozen_input_copy_hash_mismatch:{target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target_sha = sha256_file(target)
        if target_sha != source_sha:
            raise RuntimeError(f"frozen_input_copy_hash_mismatch:{target}")
    return {
        "source": str(source),
        "copy": str(target),
        "source_sha256": source_sha,
        "copy_sha256": target_sha,
    }


def _write_frozen_input_manifest(
    context: RunContext,
    config: GrpoConfig,
    tasks: list[TaskInput],
    database_evidence: list[dict[str, str]],
    *,
    command: str,
) -> dict[str, Any]:
    plan = config.path("evaluation_plan", context)
    panel = config.path("public_panel", context)
    plan_source_sha = sha256_file(plan)
    frozen_plan_path = context.run_dir / "inputs" / "frozen-plan.json"
    frozen_plan = {
        "schema_version": "t024-public-frozen-plan-v1",
        "source_plan": str(plan),
        "source_plan_sha256": plan_source_sha,
        "tasks": [
            {
                "task_id": task.task_id,
                "group_id": task.group_id,
                "public_task": task.public_task,
                "database_path": task.database_path,
                "metadata": {
                    key: value
                    for key, value in task.metadata.items()
                    if key
                    in {
                        "database_id",
                        "database_sha256",
                        "data_fingerprint",
                        "database_version",
                        "source",
                        "label_validation",
                    }
                },
            }
            for task in tasks
        ],
        "hidden_reference_boundary": (
            "No reference SQL, expected answer, or verifier-only constraint is copied."
        ),
    }
    atomic_write_json(frozen_plan_path, frozen_plan)
    plan_copy = {
        "source": str(plan),
        "copy": str(frozen_plan_path),
        "source_sha256": plan_source_sha,
        "copy_sha256": sha256_file(frozen_plan_path),
    }
    panel_copy = _copy_frozen_source(panel, context.run_dir / "inputs" / panel.name)
    rows = [
        {
            "task_id": task.task_id,
            "group_id": task.group_id,
            "public_task_sha256": canonical_hash(task.public_task),
            "database_path": task.database_path,
            "database_sha256": sha256_file(Path(task.database_path))
            if task.database_path
            else None,
            "source_metadata": {
                key: value
                for key, value in task.metadata.items()
                if key
                in {
                    "database_id",
                    "database_sha256",
                    "data_fingerprint",
                    "database_version",
                    "source",
                    "label_validation",
                }
            },
        }
        for task in tasks
    ]
    result = {
        "schema_version": "t024-frozen-input-fingerprint-v1",
        "task_ids": [task.task_id for task in tasks],
        "source_evaluation_plan": str(plan),
        "source_public_panel": str(panel),
        "plan": plan_copy,
        "public_panel": panel_copy,
        "rows": rows,
        "database_copies": database_evidence,
        "hidden_reference_boundary": (
            "TaskInput.verifier_payload is retained for the isolated verifier only; "
            "the policy/engine receives public_task."
        ),
        "training_loader_called": False,
        "command": command,
    }
    atomic_write_json(context.run_dir / "inputs" / "fingerprint-index.json", result)
    return result


def _write_frozen_protocol(
    context: RunContext,
    config: GrpoConfig,
    tasks: list[TaskInput],
    input_manifest: dict[str, Any],
    *,
    command: str,
) -> None:
    atomic_write_json(
        context.run_dir / "protocol.json",
        {
            "schema_version": "t024-frozen-protocol-v1",
            "command": command,
            "config_path": str(context.config_path),
            "config_sha256": sha256_file(context.config_path),
            "config": config.raw,
            "task_ids": [task.task_id for task in tasks],
            "input_manifest_sha256": canonical_hash(input_manifest),
            "evaluation_seed": int(config.evaluation.get("generation", {}).get("seed", 17)),
            "generation": config.evaluation.get("generation", {}),
            "runtime": config.runtime,
            "training_calls": 0,
            "hidden_reference_boundary": input_manifest["hidden_reference_boundary"],
        },
    )


def _cmd_validate(args: argparse.Namespace) -> int:
    config, context = _config(args)
    result = {
        "status": "PASS",
        "config": str(context.config_path),
        "repo_root": str(context.repo_root),
        "run_dir": str(context.run_dir),
        "model_assets_loaded": False,
        "database_calls": 0,
        "slurm_calls": 0,
        "model": {
            key: value
            for key, value in config.model.items()
            if key not in {"parent_adapter", "cache_dir"}
        },
        "budget": config.budget,
        "recommended_slurm_seconds": int(config.budget["job_hard_seconds"]),
        "asset_verification": "DEFERRED_TO_REAL_GPU_LOAD",
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _cmd_cpu_demo(args: argparse.Namespace) -> int:
    config, context = _config(args)
    result = run_online_training(
        context,
        config,
        _synthetic_tasks(),
        component_factory=_cpu_components,
        evaluation_tasks={"N": _synthetic_tasks(), "RL": _synthetic_tasks()},
        evaluation_factory=_cpu_evaluation_factory,
        run_evaluation=True,
        job_id="CPU-DEMO",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "updates": result["updates"],
                "comparison": result.get("comparison"),
                "run_dir": str(context.run_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    config, context = _config(args)
    train_ids = config.data.get("train_task_ids")
    tasks = load_task_inputs(
        config.path("train_index", context),
        repo_root=context.repo_root,
        task_ids=train_ids,
        default_database=_optional_config_path(config, context, "train_database"),
    )
    evaluation_ids = config.evaluation.get("task_ids")
    evaluation_tasks = load_task_inputs(
        config.path("evaluation_plan", context),
        repo_root=context.repo_root,
        task_ids=evaluation_ids,
        public_panel=config.path("public_panel", context),
        database_root=_optional_config_path(config, context, "evaluation_database_root"),
    )
    if len(tasks) != config.group_count:
        raise ValueError(f"train_group_count_mismatch:{len(tasks)}:{config.group_count}")
    if len(evaluation_tasks) != 4:
        raise ValueError(f"evaluation_task_count_mismatch:{len(evaluation_tasks)}:4")
    tasks, train_db_evidence = _copy_databases(tasks, context.run_dir)
    evaluation_tasks, eval_db_evidence = _copy_databases(evaluation_tasks, context.run_dir)
    _write_input_manifest(
        context, config, tasks, evaluation_tasks, train_db_evidence + eval_db_evidence
    )
    result = run_online_training(
        context,
        config,
        tasks,
        component_factory=_real_components,
        evaluation_tasks={"N": evaluation_tasks, "RL": evaluation_tasks},
        evaluation_factory=_real_evaluation_factory,
        run_evaluation=True,
        job_id=os.environ.get("SLURM_JOB_ID"),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "updates": result["updates"],
                "run_dir": str(context.run_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _unique_frozen_context(context: RunContext) -> RunContext:
    """Never turn an existing run directory into a new run by overwriting it."""

    if os.environ.get("EARL_FROZEN_RUN_RESERVED") == "1":
        return context
    base = context.run_dir
    if not base.exists() or not any(base.iterdir()):
        return context
    index = 2
    while True:
        candidate = base.parent / f"{base.name}-r{index}"
        if not candidate.exists():
            return RunContext(context.repo_root, candidate, context.config_path)
        index += 1


def _snapshot_frozen_sources(context: RunContext, label: str) -> dict[str, Any]:
    """Copy the exact allowed implementation surface into the run artifact."""

    source_root = context.run_dir / "source-snapshots" / label
    repo_root = context.repo_root
    paths = list((repo_root / "src" / "enterprise_agent_rl" / "grpo").rglob("*.py"))
    paths.extend(
        [
            repo_root / "scripts" / "training" / "run_grpo.py",
            repo_root / "scripts" / "slurm" / "grpo_online.sbatch",
            repo_root / "docs" / "grpo-training.md",
            repo_root / "configs" / "training" / "grpo-frozen-evaluation.toml",
        ]
    )
    paths.extend(sorted((repo_root / "tests").glob("test_grpo_*.py")))
    rows: list[dict[str, Any]] = []
    for source in sorted(set(paths)):
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = source.relative_to(repo_root)
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append(
            {
                "path": str(relative),
                "snapshot": str(target.relative_to(context.run_dir)),
                "sha256": sha256_file(source),
                "snapshot_sha256": sha256_file(target),
                "bytes": source.stat().st_size,
            }
        )
    atomic_write_json(
        source_root / "sha256sums.json",
        {"schema_version": "t024-source-snapshot-v1", "label": label, "files": rows},
    )
    return {"label": label, "files": rows}


def _write_frozen_asset_preflight(
    context: RunContext,
    config: GrpoConfig,
    tasks: list[TaskInput],
    database_evidence: list[dict[str, str]],
) -> dict[str, Any]:
    """Verify immutable plan, panel, DB, and adapter assets before model load."""

    evaluation = config.evaluation
    plan = config.path("evaluation_plan", context)
    panel = config.path("public_panel", context)
    expected_plan = str(evaluation.get("evaluation_plan_sha256", ""))
    expected_panel = str(evaluation.get("public_panel_sha256", ""))
    actual_plan = sha256_file(plan)
    actual_panel = sha256_file(panel)
    if expected_plan and actual_plan != expected_plan:
        raise RuntimeError(f"evaluation_plan_hash_mismatch:{actual_plan}:{expected_plan}")
    if expected_panel and actual_panel != expected_panel:
        raise RuntimeError(f"public_panel_hash_mismatch:{actual_panel}:{expected_panel}")

    from enterprise_agent_rl.benchmark import database_content_fingerprint

    expected_physical = evaluation.get("database_physical_sha256", {})
    expected_logical = evaluation.get("database_logical_fingerprint", {})
    database_rows: list[dict[str, Any]] = []
    logical_rows: dict[str, str] = {}
    for row in database_evidence:
        database_id = Path(row["source"]).stem
        expected = (
            expected_physical.get(database_id)
            if isinstance(expected_physical, dict)
            else None
        )
        if expected and row["source_sha256"] != expected:
            raise RuntimeError(
                f"validation_database_hash_mismatch:{database_id}:{row['source_sha256']}:{expected}"
            )
        logical = database_content_fingerprint(Path(row["copy"]))
        expected_logical_value = (
            expected_logical.get(database_id) if isinstance(expected_logical, dict) else None
        )
        if expected_logical_value and logical != expected_logical_value:
            raise RuntimeError(
                f"validation_database_logical_fingerprint_mismatch:{database_id}"
            )
        logical_rows[database_id] = logical
        database_rows.append(
            {
                **row,
                "database_id": database_id,
                "expected_physical_sha256": expected,
                "logical_fingerprint": logical,
                "expected_logical_fingerprint": expected_logical_value,
            }
        )

    model_cfg = config.model
    parent = Path(str(model_cfg["parent_adapter"]))
    if not parent.is_absolute():
        parent = context.repo_root / parent
    delta = Path(str(evaluation["rl_delta_adapter"]))
    if not delta.is_absolute():
        delta = context.repo_root / delta
    adapter_rows = []
    for name, root, expected_weight, expected_config in (
        (
            "parent-N",
            parent,
            str(model_cfg["parent_adapter_sha256"]),
            str(model_cfg["parent_adapter_config_sha256"]),
        ),
        (
            "fixed-RL-delta",
            delta,
            str(evaluation["rl_delta_sha256"]),
            str(evaluation["rl_delta_config_sha256"]),
        ),
    ):
        weight = root / "adapter_model.safetensors"
        adapter_config = root / "adapter_config.json"
        weight_sha = sha256_file(weight)
        config_sha = sha256_file(adapter_config)
        if weight_sha != expected_weight or config_sha != expected_config:
            raise RuntimeError(f"adapter_asset_hash_mismatch:{name}")
        adapter_rows.append(
            {
                "name": name,
                "root": str(root),
                "weight": str(weight),
                "weight_sha256": weight_sha,
                "config": str(adapter_config),
                "config_sha256": config_sha,
            }
        )

    result = {
        "schema_version": "t024-asset-preflight-v1",
        "plan": {"path": str(plan), "sha256": actual_plan, "expected_sha256": expected_plan},
        "public_panel": {
            "path": str(panel),
            "sha256": actual_panel,
            "expected_sha256": expected_panel,
        },
        "databases": database_rows,
        "database_logical_fingerprint": logical_rows,
        "adapters": adapter_rows,
        "tokenizer": {
            "status": "DEFERRED_TO_REAL_EVALUATION_MODEL_LOAD",
            "raw_sha256": str(evaluation.get("tokenizer_raw_template_sha256", "")),
            "canonical_json_sha256": str(
                evaluation.get("tokenizer_canonical_json_sha256", "")
            ),
        },
        "model": {
            "model_id": model_cfg["model_id"],
            "revision": model_cfg["revision"],
            "cache_dir": str(model_cfg["cache_dir"]),
        },
        "training_loader_called": False,
        "training_calls": 0,
    }
    atomic_write_json(context.run_dir / "asset-preflight.json", result)
    return result


def _write_cpu_checks(context: RunContext, result: dict[str, Any]) -> None:
    arms = result.get("arms", {})
    atomic_write_json(
        context.run_dir / "cpu" / "checks.json",
        {
            "schema_version": "t024-cpu-checks-v1",
            "status": result.get("status"),
            "training_loader_called": False,
            "training_calls": 0,
            "backward_calls": 0,
            "optimizer_update_calls": 0,
            "qwen_weights_loaded": False,
            "arms": {
                arm: {
                    "fresh_base": details.get("arm_metadata", {}).get("fresh_base"),
                    "parent_merge_count": details.get("arm_metadata", {}).get(
                        "parent_merge_count"
                    ),
                    "delta_merge_count": details.get("arm_metadata", {}).get(
                        "delta_merge_count"
                    ),
                    "planned": details.get("planned"),
                    "scored": details.get("scored"),
                    "unknown": details.get("unknown"),
                }
                for arm, details in arms.items()
            },
        },
    )


def _load_frozen_evaluation_tasks(config: GrpoConfig, context: RunContext) -> list[TaskInput]:
    task_ids = config.evaluation.get("task_ids")
    if not isinstance(task_ids, list) or len(task_ids) != 20:
        raise ValueError("frozen_evaluation_requires_exactly_20_task_ids")
    tasks = load_task_inputs(
        config.path("evaluation_plan", context),
        repo_root=context.repo_root,
        task_ids=[str(task_id) for task_id in task_ids],
        public_panel=config.path("public_panel", context),
        database_root=_optional_config_path(config, context, "evaluation_database_root"),
    )
    if [task.task_id for task in tasks] != [str(task_id) for task_id in task_ids]:
        raise ValueError("frozen_evaluation_task_order_mismatch")
    return tasks


def _write_command_record(context: RunContext, command: str) -> None:
    atomic_write_json(
        context.run_dir / "commands.json",
        {
            "schema_version": "t024-commands-v1",
            "command": command,
            "argv": list(sys.argv),
            "python": sys.executable,
            "cwd": str(Path.cwd()),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "recovery_elapsed_seconds": os.environ.get("EARL_FROZEN_RECOVERY_ELAPSED"),
        },
    )


def _prepare_frozen_budget_recovery(context: RunContext) -> None:
    source_value = os.environ.get("EARL_FROZEN_RECOVERY_SOURCE")
    if not source_value:
        return
    source = Path(source_value).expanduser().resolve()
    target = context.run_dir / "budget-state.json"
    if target.exists():
        raise RuntimeError(f"recovery_budget_target_already_exists:{target}")
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_job_chain(context: RunContext) -> None:
    prior = None
    if os.environ.get("EARL_FROZEN_PRIOR_JOB_ID"):
        prior = {
            "job_id": os.environ["EARL_FROZEN_PRIOR_JOB_ID"],
            "run_dir": os.environ.get("EARL_FROZEN_PRIOR_RUN_DIR"),
            "state": os.environ.get("EARL_FROZEN_PRIOR_STATE", "CANCELLED"),
            "allocation_seconds": float(
                os.environ.get("EARL_FROZEN_PRIOR_ALLOCATION_SECONDS", "0")
            ),
            "reusable_scored_results": False,
            "reason": os.environ.get(
                "EARL_FROZEN_PRIOR_REASON", "protocol_repair_before_formal_scoring"
            ),
        }
    atomic_write_json(
        context.run_dir / "job-chain.json",
        {
            "schema_version": "t024-job-chain-v1",
            "current": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "run_dir": str(context.run_dir),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
            },
            "prior": [prior] if prior is not None else [],
        },
    )


def _write_run_manifest(context: RunContext) -> None:
    rows = []
    for path in sorted(context.run_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(context.run_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    atomic_write_json(
        context.run_dir / "manifest.json",
        {
            "schema_version": "t024-manifest-v1",
            "run_dir": str(context.run_dir),
            "files": rows,
        },
    )


def _cmd_frozen_evaluation(args: argparse.Namespace, *, cpu: bool) -> int:
    config, context = _config(args)
    context = _unique_frozen_context(context)
    context.run_dir.mkdir(parents=True, exist_ok=True)
    command = "cpu-evaluate-frozen" if cpu else "evaluate-frozen"
    _prepare_frozen_budget_recovery(context)
    _write_command_record(context, command)
    _write_job_chain(context)
    tasks = _load_frozen_evaluation_tasks(config, context)
    copied_tasks, db_evidence = _copy_databases(tasks, context.run_dir)
    _snapshot_frozen_sources(context, "pre-gpu")
    _write_frozen_asset_preflight(context, config, copied_tasks, db_evidence)
    input_manifest = _write_frozen_input_manifest(
        context, config, copied_tasks, db_evidence, command=command
    )
    _write_frozen_protocol(context, config, copied_tasks, input_manifest, command=command)
    factory = _cpu_frozen_evaluation_factory if cpu else _real_evaluation_factory
    result = run_frozen_evaluation(
        context,
        config,
        copied_tasks,
        evaluation_factory=factory,
        job_id="CPU-T024-E026" if cpu else os.environ.get("SLURM_JOB_ID"),
    )
    if cpu:
        _write_cpu_checks(context, result)
    _snapshot_frozen_sources(context, "final")
    _write_run_manifest(context)
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_dir": str(context.run_dir),
                "comparison": result.get("comparison"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "COMPLETED" else 2


def _frozen_task_ids(run_dir: Path) -> list[str]:
    value = read_json(run_dir / "inputs" / "frozen-plan.json")
    tasks = value.get("tasks") if isinstance(value, dict) else None
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("frozen_plan_missing_for_finalization")
    task_ids = [str(row["task_id"]) for row in tasks if isinstance(row, dict)]
    if len(task_ids) != len(tasks) or len(task_ids) != 20 or len(set(task_ids)) != 20:
        raise RuntimeError("frozen_plan_task_ids_invalid_for_finalization")
    return task_ids


def _offline_model_evidence_check(
    run_dir: Path, trajectory_path: Path, *, arm: str
) -> dict[str, Any]:
    """Recheck persisted model evidence without executing model or verifier code."""

    trajectory = read_json(trajectory_path)
    if not isinstance(trajectory, dict):
        raise RuntimeError("offline_qualification_trajectory_not_object")
    attempts = trajectory.get("model_attempts")
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError("offline_qualification_model_attempts_missing")
    episode = trajectory.get("run_id")
    task = trajectory.get("task")
    task_id = task.get("task_id") if isinstance(task, dict) else None
    episode_id = f"{arm}-{task_id}"
    request_ids: set[str] = set()
    qualified: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise RuntimeError("offline_qualification_attempt_not_object")
        request_id = attempt.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in request_ids:
            raise RuntimeError("offline_qualification_request_id_invalid")
        request_ids.add(request_id)
        request_path = attempt.get("request_path")
        response_path = attempt.get("response_path")
        event_path = attempt.get("engine_event_path")
        if not all(
            isinstance(value, str) and value
            for value in (request_path, response_path, event_path)
        ):
            raise RuntimeError(f"offline_qualification_paths_missing:{request_id}")
        request_file = safe_relative_path(run_dir, request_path)
        response_file = safe_relative_path(run_dir, response_path)
        event_file = safe_relative_path(run_dir, event_path)
        request = read_json(request_file)
        response = read_json(response_file)
        event = read_json(event_file)
        if not all(isinstance(value, dict) for value in (request, response, event)):
            raise RuntimeError(f"offline_qualification_evidence_missing:{request_id}")
        if request.get("record_type") != "INTENT":
            raise RuntimeError(f"offline_qualification_intent_type:{request_id}")
        if response.get("record_type") != "RESPONSE":
            raise RuntimeError(f"offline_qualification_response_type:{request_id}")
        if event.get("record_type") != "ENGINE_EVENT":
            raise RuntimeError(f"offline_qualification_event_type:{request_id}")
        if event.get("completed") is not True or event.get("incomplete") is True:
            raise RuntimeError(f"offline_qualification_incomplete:{request_id}")
        expected = {
            "request_id": request_id,
            "response_id": attempt.get("response_id"),
            "episode_id": episode_id,
            "task_id": task_id,
            "stage": "evaluation",
            "arm": arm,
        }
        if not expected["response_id"]:
            raise RuntimeError(f"offline_qualification_response_id_missing:{request_id}")
        for name, expected_value in expected.items():
            if any(value.get(name) != expected_value for value in (request, response, event)):
                raise RuntimeError(f"offline_qualification_identity:{name}:{request_id}")
        if request.get("messages_sha256") != canonical_hash(request.get("messages")):
            raise RuntimeError(f"offline_qualification_messages_hash:{request_id}")
        if request.get("generation_settings_sha256") != canonical_hash(
            request.get("generation_settings")
        ):
            raise RuntimeError(f"offline_qualification_generation_hash:{request_id}")
        request_sha = sha256_file(request_file)
        response_sha = sha256_file(response_file)
        if event.get("request_sha256") != request_sha:
            raise RuntimeError(f"offline_qualification_request_file_hash:{request_id}")
        if event.get("response_sha256") != response_sha:
            raise RuntimeError(f"offline_qualification_response_file_hash:{request_id}")
        content_sha = sha256_text(response.get("content"))
        if response.get("content_sha256") != content_sha:
            raise RuntimeError(f"offline_qualification_content_hash:{request_id}")
        if event.get("content_sha256") != content_sha:
            raise RuntimeError(f"offline_qualification_event_content_hash:{request_id}")
        token_record = response.get("token_record")
        token_hash = canonical_hash(token_record)
        if response.get("token_record_sha256") != token_hash:
            raise RuntimeError(f"offline_qualification_token_hash:{request_id}")
        if event.get("token_record_sha256") != token_hash:
            raise RuntimeError(f"offline_qualification_event_token_hash:{request_id}")
        if response.get("content") != attempt.get("raw_output"):
            raise RuntimeError(f"offline_qualification_content_binding:{request_id}")
        qualified.append(
            {
                "request_id": request_id,
                "response_id": expected["response_id"],
                "request_path": request_path,
                "response_path": response_path,
                "engine_event_path": event_path,
                "request_sha256": request_sha,
                "response_sha256": response_sha,
                "content_sha256": content_sha,
                "token_record_sha256": token_hash,
                "completion_status": response.get("status"),
            }
        )
    return {
        "status": "QUALIFIED",
        "offline_revalidated": True,
        "episode_id": episode_id,
        "task_id": task_id,
        "stage": "evaluation",
        "arm": arm,
        "attempted_model_calls": len(qualified),
        "completed_model_calls": len(qualified),
        "incomplete_model_calls": 0,
        "trajectory_sha256": sha256_file(trajectory_path),
        "model_evidence": qualified,
        "trajectory_run_id": episode,
    }


def _append_missing_frozen_results(
    run_dir: Path, task_ids: list[str], *, reason: str
) -> list[dict[str, Any]]:
    appended: list[dict[str, Any]] = []
    for arm in ("Base", "N", "RL"):
        output = run_dir / "evaluation" / arm
        index = output / "verifier-results.jsonl"
        rows = read_jsonl(index)
        existing: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row.get("record_type", "RESULT") != "RESULT":
                continue
            task_id = row.get("task_id")
            if isinstance(task_id, str):
                previous = existing.get(task_id)
                if previous is not None and canonical_hash(previous) != canonical_hash(row):
                    raise RuntimeError(f"frozen_finalization_conflicting_duplicate:{arm}:{task_id}")
                existing[task_id] = row
        for task_id in task_ids:
            if task_id in existing:
                continue
            trajectory = output / "trajectories" / f"{arm}-{task_id}.json"
            row: dict[str, Any] = {
                "schema_version": "grpo-score-v1",
                "record_type": "RESULT",
                "task_id": task_id,
                "episode_id": f"{arm}-{task_id}",
                "status": "unknown",
                "unknown_reason": reason,
                "technical_unknown": True,
                "error": {
                    "type": "OfflineFinalization",
                    "message": reason,
                },
                "policy_snapshot_id": f"offline-finalized-{arm}",
                "raw_saved": trajectory.is_file(),
                "verifier_called": False,
                "source_kind": "real",
                "metadata": {"offline_finalization": True},
            }
            if trajectory.is_file():
                relative = trajectory.relative_to(run_dir)
                row["trajectory_path"] = str(relative)
                row["trajectory_sha256"] = sha256_file(trajectory)
                try:
                    row["qualification"] = _offline_model_evidence_check(
                        run_dir, trajectory, arm=arm
                    )
                except Exception as exc:
                    row["qualification"] = {
                        "status": "UNQUALIFIED",
                        "offline_revalidated": True,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
            append_jsonl(index, row)
            appended.append({"arm": arm, "task_id": task_id, "row": row})
    return appended


def _ensure_frozen_arm_manifests(
    run_dir: Path, task_ids: list[str], *, stop_reason: str
) -> None:
    base_manifest = read_json(run_dir / "evaluation" / "Base" / "arm-manifest.json")
    if not isinstance(base_manifest, dict):
        raise RuntimeError("base_arm_manifest_missing_for_finalization")
    protocol = read_json(run_dir / "protocol.json") or {}
    protocol_config = protocol.get("config", {}) if isinstance(protocol, dict) else {}
    evaluation_config = protocol_config.get("evaluation", {})
    journal_rows = read_jsonl(run_dir / "engine-evidence" / "journal.jsonl")
    for arm in ("Base", "N", "RL"):
        path = run_dir / "evaluation" / arm / "arm-manifest.json"
        if path.is_file():
            continue
        rows = read_jsonl(run_dir / "evaluation" / arm / "verifier-results.jsonl")
        scored = [row for row in rows if row.get("status") == "scored"]
        unknown = [row for row in rows if row.get("status") == "unknown"]
        successes = sum(bool(row.get("success")) for row in scored)
        identity = next(
            (
                row.get("model_identity", {})
                for row in journal_rows
                if row.get("event") == "attempt_intent_persisted" and row.get("arm") == arm
            ),
            {},
        )
        snapshot = copy.deepcopy(base_manifest["arm_snapshot"])
        metadata = copy.deepcopy(base_manifest["arm_metadata"])
        prefix = str(snapshot.get("policy_snapshot_id", "frozen")).rsplit("-", 1)[0]
        policy_snapshot_id = f"{prefix}-{arm}"
        merge_parent = int(arm in {"N", "RL"})
        merge_delta = int(arm == "RL")
        model_ref_id = identity.get("model_ref_id", f"offline-{arm}")
        model_object_id = identity.get("model_object_id", f"offline-object-{arm}")
        model_weight_fingerprint = identity.get("model_weight_fingerprint")
        semantic_delta = evaluation_config.get("rl_delta_semantic_fingerprint")
        snapshot.update(
            {
                "evaluation_arm": arm,
                "policy_snapshot_id": policy_snapshot_id,
                "model_ref_id": model_ref_id,
                "model_object_id": model_object_id,
                "model_weight_fingerprint": model_weight_fingerprint,
                "parent_merge_count": merge_parent,
                "delta_merge_count": merge_delta,
                "rl_delta_semantic_fingerprint": semantic_delta,
                "source_kind": "real",
                "training_calls": 0,
            }
        )
        metadata.update(
            {
                "evaluation_arm": arm,
                "model_ref_id": model_ref_id,
                "model_object_id": model_object_id,
                "model_weight_fingerprint": model_weight_fingerprint,
                "parent_merge_count": merge_parent,
                "delta_merge_count": merge_delta,
                "rl_delta_semantic_fingerprint": semantic_delta,
                "source_kind": "real",
                "training_calls": 0,
            }
        )
        results_path = run_dir / "evaluation" / arm / "verifier-results.jsonl"
        result = {
            "schema_version": "grpo-evaluation-v2",
            "status": "COMPLETED" if not unknown and len(scored) == len(task_ids) else "INCOMPLETE",
            "arm": arm,
            "input": {
                **copy.deepcopy(base_manifest["input"]),
                "arm": arm,
                "policy_snapshot_id": policy_snapshot_id,
                "tasks": task_ids,
            },
            "planned": len(task_ids),
            "attempted": len({row.get("task_id") for row in rows if row.get("task_id")}),
            "scored": len(scored),
            "unknown": len(unknown),
            "successes": successes,
            "stopped_before_verifier": bool(unknown),
            "stop_reason": stop_reason if unknown else None,
            "results_path": str(results_path.relative_to(run_dir)),
            "evaluation_seed": 17,
            "evaluation_mode": "greedy",
            "arm_snapshot": snapshot,
            "arm_metadata": metadata,
            "offline_finalized": True,
        }
        atomic_write_json(path, result)
        atomic_write_json(path.parent / "eval-status.json", result)


def _write_frozen_allocation_ledger(
    run_dir: Path, *, current_allocation_seconds: float | None, current_state: str | None
) -> dict[str, Any]:
    chain = read_json(run_dir / "job-chain.json") or {}
    rows: list[dict[str, Any]] = []
    for prior in chain.get("prior", []) if isinstance(chain, dict) else []:
        prior_run = Path(str(prior.get("run_dir", "")))
        wrapper = read_json(prior_run / "logs" / "wrapper-elapsed.json") if prior_run else None
        prior_wrapper_exit = (
            wrapper.get("exit_code") if isinstance(wrapper, dict) else None
        )
        rows.append(
            {
                "job_id": prior.get("job_id"),
                "state": prior.get("state"),
                "partition": "gpu4Q",
                "allocation_elapsed_seconds": prior.get("allocation_seconds"),
                "wrapper_elapsed_seconds": wrapper.get("elapsed_seconds")
                if isinstance(wrapper, dict)
                else None,
                "wrapper_exit_code": prior_wrapper_exit,
                "source": "sacct/job-chain",
            }
        )
    current_id = (chain.get("current") or {}).get("job_id") if isinstance(chain, dict) else None
    wrapper = read_json(run_dir / "logs" / "wrapper-elapsed.json")
    current_wrapper_exit = wrapper.get("exit_code") if isinstance(wrapper, dict) else None
    current_partition = (
        (chain.get("current") or {}).get("partition") if isinstance(chain, dict) else None
    )
    rows.append(
        {
            "job_id": current_id,
            "state": current_state or current_wrapper_exit,
            "partition": current_partition,
            "allocation_elapsed_seconds": current_allocation_seconds,
            "wrapper_elapsed_seconds": wrapper.get("elapsed_seconds")
            if isinstance(wrapper, dict)
            else None,
            "wrapper_exit_code": current_wrapper_exit,
            "source": "sacct/wrapper",
        }
    )
    allocation_values = [
        float(row["allocation_elapsed_seconds"])
        for row in rows
        if row.get("allocation_elapsed_seconds") is not None
    ]
    wrapper_values = [
        float(row["wrapper_elapsed_seconds"])
        for row in rows
        if row.get("wrapper_elapsed_seconds") is not None
    ]
    budget = read_json(run_dir / "budget-state.json") or {}
    result = {
        "schema_version": "t024-allocation-ledger-v1",
        "rows": rows,
        "allocation_elapsed_seconds_total": sum(allocation_values),
        "wrapper_elapsed_seconds_total": sum(wrapper_values),
        "allocation_minus_wrapper_seconds": sum(allocation_values) - sum(wrapper_values)
        if allocation_values and wrapper_values
        else None,
        "budget_consumed_gpu_seconds": budget.get("consumed_gpu_seconds"),
        "allocation_minus_budget_consumed_gpu_seconds": (
            sum(allocation_values) - float(budget["consumed_gpu_seconds"])
            if allocation_values and budget.get("consumed_gpu_seconds") is not None
            else None
        ),
        "accounting_note": (
            "Native Slurm allocation elapsed is authoritative; wrapper elapsed and the "
            "budget ledger are retained as separate diagnostics."
        ),
    }
    atomic_write_json(run_dir / "allocation-ledger.json", result)
    return result


def _copy_immutable_tree(source: Path, target: Path) -> list[dict[str, Any]]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    copied: list[dict[str, Any]] = []
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source)
        target_file = target / relative
        source_sha = sha256_file(source_file)
        if target_file.exists():
            target_sha = sha256_file(target_file)
            if target_sha != source_sha:
                raise RuntimeError(f"immutable_cpu_artifact_conflict:{target_file}")
        else:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
        copied.append(
            {
                "source": str(source_file),
                "copy": str(target_file),
                "sha256": source_sha,
                "bytes": source_file.stat().st_size,
            }
        )
    return copied


def _copy_cpu_evidence(
    run_dir: Path, *, source: str | None, log: str | None, command: str | None
) -> dict[str, Any] | None:
    if not source and not log and not command:
        return None
    result: dict[str, Any] = {"schema_version": "t024-cpu-evidence-copy-v1"}
    if source:
        result["run_files"] = _copy_immutable_tree(
            Path(source).expanduser().resolve(), run_dir / "cpu" / "reference-run"
        )
    if log:
        log_path = Path(log).expanduser().resolve()
        if not log_path.is_file():
            raise FileNotFoundError(log_path)
        target = run_dir / "cpu" / "command.stdout-stderr.txt"
        source_sha = sha256_file(log_path)
        if target.exists() and sha256_file(target) != source_sha:
            raise RuntimeError(f"immutable_cpu_log_conflict:{target}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, target)
        result["log"] = {
            "source": str(log_path),
            "copy": str(target),
            "sha256": source_sha,
            "bytes": log_path.stat().st_size,
        }
    if command:
        atomic_write_json(
            run_dir / "cpu" / "command.json",
            {
                "schema_version": "t024-cpu-command-v1",
                "command": command,
                "exit_code": 0,
                "stdout_stderr": "cpu/command.stdout-stderr.txt",
            },
        )
    atomic_write_json(run_dir / "cpu" / "evidence-copy.json", result)
    return result


def _trajectory_worker_summary(run_dir: Path, relative_path: str) -> dict[str, Any]:
    trajectory = read_json(safe_relative_path(run_dir, relative_path))
    if not isinstance(trajectory, dict):
        return {
            "sql_action_count": None,
            "sql_ok_count": None,
            "runtime_worker_sql_count": None,
            "runtime_worker_non_sql_count": None,
            "worker_count": None,
            "termination_reason": None,
        }
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return {
            "sql_action_count": None,
            "sql_ok_count": None,
            "runtime_worker_sql_count": None,
            "runtime_worker_non_sql_count": None,
            "worker_count": None,
            "termination_reason": trajectory.get("terminal_reason"),
        }
    sql_steps = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("tool_name") == "sql"
    ]
    non_sql_steps = [
        step
        for step in steps
        if isinstance(step, dict)
        and step.get("tool_name") in {"schema", "dictionary", "check"}
    ]

    def _is_budget_rejection(step: dict[str, Any]) -> bool:
        result = step.get("tool_result")
        return isinstance(result, dict) and "budget" in str(result.get("error") or "").lower()

    def _is_ok(step: dict[str, Any]) -> bool:
        result = step.get("tool_result")
        return isinstance(result, dict) and bool(result.get("ok"))

    sql_worker = sum(not _is_budget_rejection(step) for step in sql_steps)
    non_sql_worker = sum(not _is_budget_rejection(step) for step in non_sql_steps)
    return {
        "sql_action_count": len(sql_steps),
        "sql_ok_count": sum(_is_ok(step) for step in sql_steps),
        "runtime_worker_sql_count": sql_worker,
        "runtime_worker_non_sql_count": non_sql_worker,
        "worker_count": sql_worker + non_sql_worker,
        "termination_reason": trajectory.get("terminal_reason"),
    }


def _write_frozen_arm_audits(
    run_dir: Path, task_ids: list[str], metrics: dict[str, Any]
) -> None:
    """Write compact per-task worker and qualification indexes for review."""

    for arm in ("Base", "N", "RL"):
        index = run_dir / "evaluation" / arm / "verifier-results.jsonl"
        formal_rows = {
            str(row.get("task_id")): row
            for row in read_jsonl(index)
            if row.get("record_type", "RESULT") == "RESULT"
        }
        aggregate_rows = {
            str(row.get("task_id")): row for row in metrics[arm].get("entries", [])
        }
        worker_rows: list[dict[str, Any]] = []
        qualification_rows: list[dict[str, Any]] = []
        for task_id in task_ids:
            formal = formal_rows[task_id]
            entry = aggregate_rows.get(task_id, {})
            trajectory_path = formal.get("trajectory_path")
            trajectory_summary = (
                _trajectory_worker_summary(run_dir, trajectory_path)
                if isinstance(trajectory_path, str)
                else {}
            )
            sql = entry.get("sql", {})
            worker_rows.append(
                {
                    "task_id": task_id,
                    "status": formal.get("status"),
                    "technical_unknown": bool(formal.get("technical_unknown", False)),
                    "verifier_called": bool(formal.get("verifier_called", False)),
                    "trajectory_path": trajectory_path,
                    "trajectory_sha256": formal.get("trajectory_sha256"),
                    "sql_action_count": entry.get(
                        "sql_action_count",
                        sql.get("sql_action_count", trajectory_summary.get("sql_action_count")),
                    ),
                    "sql_ok_count": entry.get(
                        "sql_ok_count",
                        sql.get("sql_ok_count", trajectory_summary.get("sql_ok_count")),
                    ),
                    "runtime_worker_sql_count": formal.get(
                        "runtime_worker_sql_count",
                        entry.get(
                            "runtime_worker_sql_count",
                            trajectory_summary.get("runtime_worker_sql_count"),
                        ),
                    ),
                    "runtime_worker_non_sql_count": formal.get(
                        "runtime_worker_non_sql_count",
                        entry.get(
                            "runtime_worker_non_sql_count",
                            trajectory_summary.get("runtime_worker_non_sql_count"),
                        ),
                    ),
                    "worker_count": trajectory_summary.get("worker_count"),
                    "verifier_sql_count": formal.get(
                        "verifier_sql_count", entry.get("verifier_sql_count")
                    ),
                    "termination_reason": trajectory_summary.get("termination_reason"),
                    "audit_source": (
                        "verifier-result"
                        if formal.get("status") == "scored"
                        else "trajectory-only-before-verifier"
                    ),
                }
            )
            qualification = formal.get("qualification") or {}
            qualification_rows.append(
                {
                    "task_id": task_id,
                    "status": formal.get("status"),
                    "qualification_status": qualification.get("status"),
                    "offline_revalidated": qualification.get("offline_revalidated", False),
                    "attempted_model_calls": qualification.get("attempted_model_calls"),
                    "completed_model_calls": qualification.get("completed_model_calls"),
                    "incomplete_model_calls": qualification.get("incomplete_model_calls"),
                    "trajectory_path": trajectory_path,
                    "trajectory_sha256": formal.get("trajectory_sha256"),
                    "verifier_called": bool(formal.get("verifier_called", False)),
                    "unknown_reason": formal.get("unknown_reason"),
                }
            )
        atomic_write_json(
            run_dir / "evaluation" / arm / "worker-audit.json",
            {
                "schema_version": "t024-worker-audit-v1",
                "arm": arm,
                "planned": len(task_ids),
                "rows": worker_rows,
                "unknown_rows_do_not_contribute_business_success": True,
            },
        )
        atomic_write_json(
            run_dir / "evaluation" / arm / "qualification-audit.json",
            {
                "schema_version": "t024-qualification-audit-v1",
                "arm": arm,
                "planned": len(task_ids),
                "rows": qualification_rows,
            },
        )


def _write_frozen_report(
    run_dir: Path,
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    appended: list[dict[str, Any]],
    allocation: dict[str, Any],
) -> None:
    budget = read_json(run_dir / "budget-state.json") or {}
    status = read_json(run_dir / "run-status.json") or {}
    commands = read_json(run_dir / "commands.json") or {}
    counters_json = json.dumps(
        budget.get("call_counters", {}), ensure_ascii=False, sort_keys=True
    )
    allocation_json = json.dumps(allocation, ensure_ascii=False, sort_keys=True)
    lines = [
        "# T-024 / E-026 Frozen Base/SFT/RL Evaluation",
        "",
        (
            "Status: **INCOMPLETE_EVALUATION**. The run reached the cumulative "
            "verifier-call ceiling; missing evidence is reported as technical "
            "unknown and is not converted to a business failure or a reduced "
            "denominator."
        ),
        "",
        "## Frozen contract",
        "",
        (
            "- Arms: Base, SFT N, and RL=N plus the fixed external delta; fresh "
            "load order Base → N → RL."
        ),
        (
            "- Panel: validation-0 then validation-1, 10 tasks each, fixed order "
            "from `inputs/frozen-plan.json`; planned denominator is 20 per arm."
        ),
        (
            "- Generation: greedy T=0, seed 17, top_p=1, top_k=0, repetition "
            "penalty 1, max 512, json_fence, format retry 1."
        ),
        (
            "- Training/backward/optimizer/update calls: 0. Hidden reference "
            "SQL/answers were retained only for the isolated Verifier."
        ),
        "",
        "## Per-arm aggregation",
        "",
        (
            "| arm | planned | attempted | qualified | scored | unknown | successes "
            "| success rate | model calls attempted/completed | Verifier calls "
            "| SQL Action / SQL_OK | Runtime worker SQL / non-SQL | Verifier SQL |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("Base", "N", "RL"):
        value = metrics[arm]
        rate = value.get("success_rate")
        cells = [
            arm,
            value.get("planned"),
            value.get("attempted"),
            value.get("qualified"),
            value.get("scored"),
            value.get("unknown"),
            value.get("successes"),
            rate if rate is not None else "null",
            f"{value.get('attempted_model_calls')}/{value.get('completed_model_calls')}",
            value.get("verifier_called"),
            f"{value.get('sql_action_count')} / {value.get('sql_ok_count')}",
            f"{value.get('runtime_worker_sql_count')} / "
            f"{value.get('runtime_worker_non_sql_count')}",
            value.get("verifier_sql_count"),
        ]
        lines.append(
            "| " + " | ".join(str(cell) for cell in cells) + " |"
        )
    lines.extend(
        [
            "",
            (
                "The RL `validation-1-monthly` trajectory has complete, "
                "offline-revalidated model evidence but no Verifier result "
                "because the inherited old-protocol call consumed one of the "
                "60 cumulative Verifier calls; the current run then rejected "
                "the 61st call. It is recorded in the RL index as "
                "`technical_unknown`."
            ),
            "",
            "## Comparisons",
            "",
            (
                f"- Comparison status: `{comparison.get('status')}`; fixed "
                f"denominator: `{comparison.get('denominator')}` (null until "
                "all three arms are 20/20 qualified and scored)."
            ),
            (
                "- RL−N, N−Base, and RL−Base point estimates, positive transfers, "
                "and negative transfers: null/empty because the fixed 20-task "
                "paired comparison is incomplete."
            ),
            (
                "- The complete per-task pair table, including the explicit RL "
                "monthly unknown, is in `comparison.json`."
            ),
            "",
            "## Budget and Slurm accounting",
            "",
            (
                f"- Job chain: `{commands.get('job_id')}` current; prior "
                "protocol-aborted Job `20419283` is preserved and explicitly "
                "non-reusable; corrected Job `20419292` exited with the budget "
                "stop."
            ),
            (
                f"- Budget counters: `{counters_json}`; consumed GPU seconds "
                f"`{budget.get('consumed_gpu_seconds')}` including the prior "
                "attempt."
            ),
            f"- Allocation ledger: `{allocation_json}`.",
            (
                "- The first Job used inherited HF sampling configuration and "
                "was cancelled before formal scoring; its evidence remains "
                "untouched and is excluded from this run."
            ),
            "",
            "## Evidence and closeout",
            "",
            (
                f"- Offline finalization records `{len(appended)}` explicit "
                "post-run result row(s); no existing request, response, engine "
                "event, trajectory, or score row was replaced."
            ),
            (
                f"- The raw execution status is preserved in `run-status.json`: "
                f"`{status.get('status')}` with `{status.get('error')}`."
            ),
            (
                "- Each completed model attempt is bound by request/response/"
                "engine-event IDs, content/token hashes, stage/arm/episode "
                "identity, and append-only journal rows. The incomplete "
                "business score is not inferred from the model trajectory."
            ),
            (
                "- Source snapshots `source-snapshots/pre-gpu`, "
                "`source-snapshots/final`, and `source-snapshots/post-finalize` "
                "separate the executed source from the final audited source."
            ),
            "",
            "## Verification commands",
            "",
            (
                "- CPU: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                "PYTHONHASHSEED=42 .venv-gpu/bin/python scripts/training/run_grpo.py "
                "cpu-evaluate-frozen ...` → exit 0; 22 related pytest tests passed; "
                "Ruff, Python compilation, and Bash syntax checks passed."
            ),
            (
                "- The CPU run bundle and combined stdout/stderr are copied under "
                "`cpu/` and are included in the final manifest."
            ),
            (
                "- GPU: corrected Job `20419292`, native Slurm allocation elapsed "
                "2412 seconds, V100-SXM2-32GB; wrapper exit code 1 is the expected "
                "nonzero budget-stop closeout."
            ),
            "",
            (
                "See `metrics.json`, `comparison.json`, `allocation-ledger.json`, "
                "`asset-preflight.json`, `engine-evidence/journal.jsonl`, and "
                "each `evaluation/*/verifier-results.jsonl` for machine-readable "
                "evidence."
            ),
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cmd_finalize_frozen(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    task_ids = _frozen_task_ids(run_dir)
    reason = str(args.unknown_reason)
    appended = _append_missing_frozen_results(run_dir, task_ids, reason=reason)
    offline_unknowns: list[dict[str, Any]] = []
    for arm in ("Base", "N", "RL"):
        index = run_dir / "evaluation" / arm / "verifier-results.jsonl"
        for row in read_jsonl(index):
            metadata = row.get("metadata")
            if isinstance(metadata, dict) and metadata.get("offline_finalization"):
                offline_unknowns.append(
                    {
                        "arm": arm,
                        "task_id": row.get("task_id"),
                        "status": row.get("status"),
                        "unknown_reason": row.get("unknown_reason"),
                    }
                )
    _ensure_frozen_arm_manifests(run_dir, task_ids, stop_reason=reason)
    metrics: dict[str, Any] = {}
    for arm in ("Base", "N", "RL"):
        index = run_dir / "evaluation" / arm / "verifier-results.jsonl"
        rows = read_jsonl(index)
        observed = {
            str(row.get("task_id"))
            for row in rows
            if row.get("record_type", "RESULT") == "RESULT"
        }
        if observed != set(task_ids):
            raise RuntimeError(f"frozen_finalization_task_set_mismatch:{arm}")
        metrics[arm] = aggregate_source(
            source_root=run_dir,
            verifier_index=index,
            identity_field="task_id",
            expected_count=len(task_ids),
            source_kind=f"T024-E026-real-validation-{arm}",
        )
    comparison = three_arm_comparison(
        metrics["Base"], metrics["N"], metrics["RL"], expected_count=len(task_ids)
    )
    atomic_write_json(run_dir / "metrics.json", metrics)
    atomic_write_json(run_dir / "comparison.json", comparison)
    _write_frozen_arm_audits(run_dir, task_ids, metrics)
    allocation = _write_frozen_allocation_ledger(
        run_dir,
        current_allocation_seconds=args.current_allocation_seconds,
        current_state=args.current_state,
    )
    _copy_cpu_evidence(
        run_dir,
        source=args.cpu_source,
        log=args.cpu_log,
        command=args.cpu_command,
    )
    atomic_write_json(
        run_dir / "finalization.json",
        {
            "schema_version": "t024-frozen-finalization-v1",
            "unknown_reason": reason,
            "newly_appended": appended,
            "offline_unknowns": offline_unknowns,
            "task_ids": task_ids,
            "comparison_status": comparison["status"],
            "allocation_ledger": "allocation-ledger.json",
        },
    )
    context = RunContext(
        repo_root=Path(args.repo_root).expanduser().resolve(),
        run_dir=run_dir,
        config_path=Path(args.config).expanduser().resolve(),
    )
    _snapshot_frozen_sources(context, "post-finalize")
    _write_frozen_report(run_dir, metrics, comparison, offline_unknowns, allocation)
    finalization_exit = 0 if comparison["status"] == "COMPLETE" else 2
    atomic_write_json(
        run_dir / "finalization-command.json",
        {
            "schema_version": "t024-finalization-command-v1",
            "argv": list(sys.argv),
            "cwd": str(Path.cwd()),
            "exit_code": finalization_exit,
        },
    )
    _write_run_manifest(context)
    print(
        json.dumps(
            {
                "status": comparison["status"],
                "run_dir": str(run_dir),
                "appended_unknowns": len(appended),
                "offline_unknowns": len(offline_unknowns),
                "metrics": "metrics.json",
                "comparison": "comparison.json",
            },
            ensure_ascii=False,
        )
    )
    return finalization_exit


def _cmd_finalize(args: argparse.Namespace) -> int:
    root = Path(args.source_root).expanduser().resolve()
    index = Path(args.verifier_index).expanduser()
    if not index.is_absolute():
        index = root / index
    output = Path(args.output).expanduser().resolve()
    result = aggregate_source(
        source_root=root,
        verifier_index=index,
        identity_field=args.identity_field,
        expected_count=int(args.expected_count),
        source_kind=args.source_kind,
        fixture=bool(args.fixture),
    )
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "status": result["evidence_status"],
                "scored": result["scored"],
                "unknown": result["unknown"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["unknown"] == 0 else 2


def _cmd_replay(args: argparse.Namespace) -> int:
    result = replay_t021(Path(args.t021_root), Path(args.output))
    print(
        json.dumps(
            {
                "status": result["status"],
                "N": {
                    key: result["N"].get(key)
                    for key in (
                        "scored",
                        "unknown",
                        "successes",
                        "sql_action_count",
                        "sql_ok_count",
                    )
                },
                "RL": {
                    key: result["RL"].get(key)
                    for key in (
                        "scored",
                        "unknown",
                        "successes",
                        "sql_action_count",
                        "sql_ok_count",
                    )
                },
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "PASS" else 2


def _cmd_score_saved(args: argparse.Namespace) -> int:
    result = score_saved_trajectory(
        source_run=Path(args.source_run).expanduser().resolve(),
        private_task_plan=Path(args.private_task_plan).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        repo_root=Path(args.repo_root).expanduser().resolve(),
        arm=str(args.arm),
        task_id=str(args.task_id),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "supplement_status": result["supplement_status"],
                "run_dir": result["run_dir"],
                "reused": result["reused"],
                "new_verifier_calls": result["new_verifier_calls"],
                "new_verifier_sql": result["new_verifier_sql"],
                "comparison": result["comparison"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["supplement_status"] == "scored" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_grpo.py", description="Reusable two-round online GRPO entrypoint"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repo-root", required=True, help="absolute repository root")
        command.add_argument("--config", required=True, help="explicit TOML config path")
        command.add_argument("--run", required=True, help="explicit durable run directory")

    validate = sub.add_parser(
        "validate-config", help="validate configuration without model/database access"
    )
    common(validate)
    validate.set_defaults(handler=_cmd_validate)
    cpu = sub.add_parser("cpu-demo", help="run the injected CPU fake/tiny integration")
    common(cpu)
    cpu.set_defaults(handler=_cmd_cpu_demo)
    train = sub.add_parser("train", help="run the real one-GPU online loop")
    common(train)
    train.set_defaults(handler=_cmd_train)
    evaluate = sub.add_parser(
        "evaluate-frozen", help="run Base/N/RL frozen validation without the train stack"
    )
    common(evaluate)
    evaluate.set_defaults(handler=lambda args: _cmd_frozen_evaluation(args, cpu=False))
    cpu_evaluate = sub.add_parser(
        "cpu-evaluate-frozen", help="run the strict three-arm evaluation contract with CPU fakes"
    )
    common(cpu_evaluate)
    cpu_evaluate.set_defaults(handler=lambda args: _cmd_frozen_evaluation(args, cpu=True))
    finalize_frozen = sub.add_parser(
        "finalize-frozen",
        help="close an interrupted frozen run without executing model or verifier code",
    )
    common(finalize_frozen)
    finalize_frozen.add_argument(
        "--unknown-reason", default="post_run_missing_result_after_budget_stop"
    )
    finalize_frozen.add_argument("--current-allocation-seconds", type=float, default=None)
    finalize_frozen.add_argument("--current-state", default=None)
    finalize_frozen.add_argument("--cpu-source", default=None)
    finalize_frozen.add_argument("--cpu-log", default=None)
    finalize_frozen.add_argument("--cpu-command", default=None)
    finalize_frozen.set_defaults(handler=_cmd_finalize_frozen)
    finalize = sub.add_parser("finalize", help="aggregate one explicit formal verifier index")
    finalize.add_argument("--source-root", required=True)
    finalize.add_argument("--verifier-index", required=True)
    finalize.add_argument("--identity-field", choices=("task_id", "episode_id"), default="task_id")
    finalize.add_argument("--expected-count", required=True, type=int)
    finalize.add_argument("--source-kind", required=True)
    finalize.add_argument("--fixture", action="store_true")
    finalize.add_argument("--output", required=True)
    finalize.set_defaults(handler=_cmd_finalize)
    replay = sub.add_parser("replay-t021", help="offline replay of delivered T-021 formal evidence")
    replay.add_argument("--t021-root", required=True)
    replay.add_argument("--output", required=True)
    replay.set_defaults(handler=_cmd_replay)
    score_saved = sub.add_parser(
        "score-saved",
        help="CPU-only production Verifier scoring for one explicit saved trajectory",
    )
    score_saved.add_argument("--repo-root", required=True)
    score_saved.add_argument("--source-run", required=True)
    score_saved.add_argument("--private-task-plan", required=True)
    score_saved.add_argument("--arm", required=True)
    score_saved.add_argument("--task-id", required=True)
    score_saved.add_argument("--output", required=True)
    score_saved.set_defaults(handler=_cmd_score_saved)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
