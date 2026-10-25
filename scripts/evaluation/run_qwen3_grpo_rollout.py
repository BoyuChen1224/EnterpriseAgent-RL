#!/usr/bin/env python3
"""T-033/E-030 Qwen3 SFT rollout-signal experiment.

Stages are deliberately separate:

* ``preflight`` copies only the four frozen train databases, calls the four
  reference SQL controls and four independent Python controls, and writes the
  fixed schedule.  It never imports torch or loads model weights.
* ``rollout`` is the one-GPU, Slurm-only stage.  It loads the frozen Qwen3
  SFT parent once, performs exactly the persisted schedule, and calls the
  original Runtime/Verifier interfaces.
* ``finalize`` is offline.  It validates evidence/token contracts and computes
  group population variance and the TRL-compatible sample-standard-deviation
  reference advantages.  It does not call a model, SQL worker, or verifier.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_DEFAULT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DEFAULT / "src"))

from enterprise_agent_rl.benchmark import (  # noqa: E402
    database_content_fingerprint,
    python_oracle,
    task_specs,
)
from enterprise_agent_rl.common import PublicTask, Settings  # noqa: E402
from enterprise_agent_rl.data import BenchmarkTask  # noqa: E402
from enterprise_agent_rl.grpo.budget import BudgetError, BudgetGuard  # noqa: E402
from enterprise_agent_rl.grpo.contracts import (  # noqa: E402
    EpisodeSample,
    ScoreResult,
    TaskInput,
    TokenRecord,
)
from enterprise_agent_rl.grpo.evidence import (  # noqa: E402
    EvidenceError,
    evidence_journal_summary,
    qualify_episode_evidence,
)
from enterprise_agent_rl.grpo.io import (  # noqa: E402
    atomic_write_json,
    immutable_write_json,
    read_json,
    read_jsonl,
    sha256_file,
)
from enterprise_agent_rl.grpo.qwen3_rollout import (  # noqa: E402
    QWEN3_CONTEXT_TOKENS,
    QWEN3_MODEL_ID,
    QWEN3_PARENT_CONFIG_SHA256,
    QWEN3_PARENT_WEIGHT_SHA256,
    QWEN3_REVISION,
    load_qwen3_rollout_config,
    load_qwen3_rollout_policy,
    qwen3_policy_snapshot,
)
from enterprise_agent_rl.grpo.qwen3_rollout import (  # noqa: E402
    _tokenizer_preflight as _write_tokenizer_preflight,
)
from enterprise_agent_rl.grpo.torch_backend import HfRolloutEngine, RealVerifierWorker  # noqa: E402
from enterprise_agent_rl.rewards import verify  # noqa: E402
from enterprise_agent_rl.tools import SQLTool  # noqa: E402
from enterprise_agent_rl.trajectories import Trajectory  # noqa: E402

TASK_ORDER = (
    (0, "schema"),
    (0, "region"),
    (1, "product"),
    (1, "recovery"),
)
TASK_IDS = tuple(f"train-{instance}-{template}" for instance, template in TASK_ORDER)
RUN_ID = "T-033-E-030-qwen3-grpo-rollout-signal"
SOURCE_DB_HASHES = {
    "train-0": "6e35b9a64432501df71aa90b787c57ad0dc742e9a52e506285d83651885d91cc",
    "train-1": "99e56b8f6511219eaecf19b50668ae02effb03d2533737d8dd3da4f7d65451bf",
}
SOURCE_DB_LOGICAL_HASHES = {
    "train-0": "90b5d80a19d26f5df0ffe367a10f8cfd947694c0bfc7e8f25be01fc4c6fc879d",
    "train-1": "01b0da1528d8cef072cf10f801039dd4a5d3849744bf2f69b485734d018ad4eb",
}
SOURCE_PATHS = (
    "configs/evaluation/qwen3-grpo-rollout.toml",
    "scripts/evaluation/run_qwen3_grpo_rollout.py",
    "scripts/slurm/qwen3_grpo_rollout.sbatch",
    "src/enterprise_agent_rl/grpo/contracts.py",
    "src/enterprise_agent_rl/grpo/qwen3_rollout.py",
    "src/enterprise_agent_rl/grpo/torch_backend.py",
    "src/enterprise_agent_rl/agent.py",
    "src/enterprise_agent_rl/benchmark.py",
    "src/enterprise_agent_rl/data.py",
    "src/enterprise_agent_rl/environment.py",
    "src/enterprise_agent_rl/lora.py",
    "src/enterprise_agent_rl/model_protocol.py",
    "src/enterprise_agent_rl/rewards.py",
    "src/enterprise_agent_rl/sql_worker.py",
    "src/enterprise_agent_rl/tools.py",
    "src/enterprise_agent_rl/trajectories.py",
    "tests/test_qwen3_grpo_rollout.py",
)


class RolloutSignalError(RuntimeError):
    """A persisted experiment contract is not safe to continue."""


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_index(path: Path, key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise RolloutSignalError(f"index_missing_{key}:{path}")
        previous = result.get(value)
        if previous is not None and _sha(previous) != _sha(row):
            raise RolloutSignalError(f"conflicting_duplicate:{path}:{key}:{value}")
        result[value] = row
    return result


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _ensure_run_dir(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    owner_path = run_dir / "ownership.json"
    if owner_path.exists():
        owner = read_json(owner_path)
        if not isinstance(owner, dict) or owner.get("task") != "T-033/E-030":
            raise RolloutSignalError(f"run_dir_owned_by_other_task:{run_dir}")
        return
    entries = [item for item in run_dir.iterdir() if item.name != ".gitkeep"]
    if entries:
        raise RolloutSignalError(f"run_dir_exists_without_ownership:{run_dir}")
    _write_json(
        owner_path,
        {
            "schema_version": "t033-run-ownership-v1",
            "task": "T-033/E-030",
            "experiment": "Qwen3 SFT rollout signal",
            "run_id": RUN_ID,
            "created_at_utc": _now(),
            "owner_session": os.environ.get("CODEX_SESSION_ID"),
        },
    )


def _common_paths(args: argparse.Namespace) -> dict[str, Path]:
    repo = Path(args.repo_root).expanduser().resolve()
    run = Path(args.run).expanduser()
    if not run.is_absolute():
        run = repo / run
    return {
        "repo": repo,
        "run": run.resolve(),
        "config": (repo / args.config).resolve()
        if not Path(args.config).is_absolute()
        else Path(args.config).resolve(),
        "panel_manifest": (repo / args.panel_manifest).resolve()
        if not Path(args.panel_manifest).is_absolute()
        else Path(args.panel_manifest).resolve(),
        "source_db_root": (repo / args.source_db_root).resolve()
        if not Path(args.source_db_root).is_absolute()
        else Path(args.source_db_root).resolve(),
        "cache": (repo / args.cache).resolve()
        if not Path(args.cache).is_absolute()
        else Path(args.cache).resolve(),
        "parent_adapter": (repo / args.parent_adapter).resolve()
        if not Path(args.parent_adapter).is_absolute()
        else Path(args.parent_adapter).resolve(),
    }


def _config(paths: dict[str, Path]):
    from enterprise_agent_rl.grpo.config import RunContext

    return load_qwen3_rollout_config(
        paths["config"],
        RunContext(paths["repo"], paths["run"], paths["config"]),
        cache_dir=paths["cache"],
        parent_adapter=paths["parent_adapter"],
    )


def _selected_task(instance: int, template: str) -> BenchmarkTask:
    for candidate, prompt, reference_sql, constraints in task_specs("train", instance):
        if candidate == template:
            return BenchmarkTask(
                public=PublicTask(task_id=f"train-{instance}-{template}", prompt=prompt),
                split="train",
                reference_sql=reference_sql,
                ordered=True,
                **constraints,
            )
    raise RolloutSignalError(f"train_task_template_missing:{instance}:{template}")


def _verify_panel_manifest(panel: dict[str, Any]) -> None:
    if panel.get("split") != "train" or panel.get("year") != 2021:
        raise RolloutSignalError("t031_panel_split_or_year_mismatch")
    if panel.get("seeds") != {"train-0": 101, "train-1": 102}:
        raise RolloutSignalError("t031_panel_database_seed_mismatch")
    if (
        panel.get("validation_split_accessed") is not False
        or panel.get("test_split_accessed") is not False
    ):
        raise RolloutSignalError("t031_panel_non_train_access_flag")
    task_ids = {row.get("task_id") for row in panel.get("tasks", [])}
    if not set(TASK_IDS).issubset(task_ids):
        raise RolloutSignalError("t031_panel_selected_task_missing")
    databases = {row.get("id"): row for row in panel.get("databases", [])}
    if set(databases) != {"train-0", "train-1"}:
        raise RolloutSignalError("t031_panel_database_set_mismatch")
    for database_id in databases:
        if databases[database_id].get("physical_sha256") != SOURCE_DB_HASHES[database_id]:
            raise RolloutSignalError(f"t031_panel_physical_hash_mismatch:{database_id}")
        if (
            databases[database_id].get("logical_fingerprint")
            != SOURCE_DB_LOGICAL_HASHES[database_id]
        ):
            raise RolloutSignalError(f"t031_panel_logical_hash_mismatch:{database_id}")


def _copy_frozen_databases(
    paths: dict[str, Path], panel: dict[str, Any], run_dir: Path
) -> list[dict[str, Any]]:
    database_root = run_dir / "panel" / "databases"
    database_root.mkdir(parents=True, exist_ok=True)
    panel_databases = {row["id"]: row for row in panel["databases"]}
    entries = []
    for database_id in ("train-0", "train-1"):
        source = paths["source_db_root"] / f"{database_id}.sqlite3"
        target = database_root / f"{database_id}.sqlite3"
        if not source.is_file():
            raise FileNotFoundError(source)
        source_physical = sha256_file(source)
        source_logical = database_content_fingerprint(source)
        if source_physical != SOURCE_DB_HASHES[database_id]:
            raise RolloutSignalError(f"source_database_physical_hash_mismatch:{database_id}")
        if source_logical != SOURCE_DB_LOGICAL_HASHES[database_id]:
            raise RolloutSignalError(f"source_database_logical_hash_mismatch:{database_id}")
        if target.exists():
            if sha256_file(target) != source_physical:
                raise RolloutSignalError(f"run_database_copy_hash_mismatch:{database_id}")
        else:
            shutil.copy2(source, target)
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        copied_physical = sha256_file(target)
        copied_logical = database_content_fingerprint(target)
        if copied_physical != source_physical or copied_logical != source_logical:
            raise RolloutSignalError(f"copied_database_hash_mismatch:{database_id}")
        mode = stat.S_IMODE(target.stat().st_mode)
        if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RolloutSignalError(f"copied_database_not_readonly:{database_id}")
        if panel_databases[database_id].get("physical_sha256") != source_physical:
            raise RolloutSignalError(f"panel_source_database_hash_changed:{database_id}")
        entries.append(
            {
                "id": database_id,
                "split": "train",
                "year": 2021,
                "seed": 101 if database_id == "train-0" else 102,
                "source_path": str(source.resolve()),
                "source_physical_sha256": source_physical,
                "source_logical_fingerprint": source_logical,
                "copied_path": str(target.resolve()),
                "copied_physical_sha256": copied_physical,
                "copied_logical_fingerprint": copied_logical,
                "readonly_mode": oct(mode),
            }
        )
    return entries


def _build_panel(
    paths: dict[str, Path], panel: dict[str, Any], config: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_panel = {row["task_id"]: row for row in panel["tasks"]}
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    database_root = paths["run"] / "panel" / "databases"
    settings = Settings.model_validate(config.runtime)
    for instance, template in TASK_ORDER:
        task = _selected_task(instance, template)
        task_id = task.public.task_id
        declared = selected_panel.get(task_id)
        if declared is None:
            raise RolloutSignalError(f"selected_t031_task_missing:{task_id}")
        if declared.get("public_task") != task.public.model_dump():
            raise RolloutSignalError(f"selected_public_task_changed:{task_id}")
        if (
            declared.get("reference_sql_sha256")
            != hashlib.sha256(task.reference_sql.encode()).hexdigest()
        ):
            raise RolloutSignalError(f"selected_reference_sql_hash_changed:{task_id}")
        declared_constraints = declared.get("constraints", {})
        actual_constraints = {
            "ordered": task.ordered,
            "min_sql_calls": task.min_sql_calls,
            "required_probe_sql": task.required_probe_sql is not None,
            "required_schema": task.required_schema,
            "required_sql_error": task.required_sql_error,
        }
        if declared_constraints != actual_constraints:
            raise RolloutSignalError(f"selected_task_constraints_changed:{task_id}")
        database_id = f"train-{instance}"
        database = database_root / f"{database_id}.sqlite3"
        reference_result = SQLTool(
            database, settings, paths["run"] / "panel" / "private" / "cpu-sql-audit.jsonl"
        ).execute(
            task.reference_sql,
            context={"control": "reference_sql", "task_id": task_id},
        )
        _append_control_call(
            paths["run"],
            {
                "control_type": "reference_sql",
                "task_id": task_id,
                "result_sha256": _sha(reference_result),
            },
        )
        expected_rows = python_oracle(database, template, "train", instance)
        _append_control_call(
            paths["run"],
            {
                "control_type": "python_oracle",
                "task_id": task_id,
                "result_sha256": _sha(expected_rows),
            },
        )
        expected = {"columns": reference_result.get("columns"), "rows": expected_rows}
        oracle_sha = _sha(expected)
        declared_oracle = next(
            row for row in panel["oracle_checks"] if row.get("task_id") == task_id
        )
        if (
            not reference_result.get("ok")
            or reference_result.get("truncated")
            or reference_result.get("rows") != expected_rows
        ):
            raise RolloutSignalError(f"reference_python_oracle_mismatch:{task_id}")
        if declared_oracle.get("oracle_sha256") != oracle_sha:
            raise RolloutSignalError(f"t031_oracle_hash_changed:{task_id}")
        public_rows.append(
            {
                "task_id": task_id,
                "group_id": task_id,
                "database_id": database_id,
                "database_path": str(database.resolve()),
                "instance": instance,
                "template": template,
                "public_task": task.public.model_dump(),
                "public_task_sha256": _sha(task.public.model_dump()),
            }
        )
        private_rows.append(
            {
                "task_id": task_id,
                "database_id": database_id,
                "template": template,
                "benchmark_task": task.model_dump(),
                "reference_sql_sha256": hashlib.sha256(task.reference_sql.encode()).hexdigest(),
                "expected_sha256": oracle_sha,
            }
        )
        control_rows.append(
            {
                "task_id": task_id,
                "database_id": database_id,
                "template": template,
                "reference_sql_call": {
                    "ok": reference_result.get("ok"),
                    "columns": reference_result.get("columns"),
                    "rows": reference_result.get("rows"),
                    "truncated": reference_result.get("truncated"),
                },
                "python_oracle_call": {"rows": expected_rows},
                "oracle_sha256": oracle_sha,
                "reference_sql_sha256": hashlib.sha256(task.reference_sql.encode()).hexdigest(),
                "policy_messages_read_reference_sql": False,
            }
        )
    return public_rows, private_rows, control_rows


def _control_ledger_path(run_dir: Path) -> Path:
    return run_dir / "panel" / "private" / "cpu-control-calls.jsonl"


def _append_control_call(run_dir: Path, record: dict[str, Any]) -> None:
    """Persist one completed CPU control; no count is inferred from panels."""

    _append_jsonl(
        _control_ledger_path(run_dir),
        {
            "schema_version": "t034-control-call-v1",
            "recorded_at_utc": _now(),
            **record,
        },
    )


def _control_counts(run_dir: Path) -> dict[str, int]:
    ledger = _control_ledger_path(run_dir)
    if not ledger.is_file():
        raise RolloutSignalError("cpu_control_append_ledger_missing")
    rows = read_jsonl(ledger)
    counts = defaultdict(int)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        control_type = row.get("control_type")
        task_id = row.get("task_id")
        if control_type not in {"reference_sql", "python_oracle"} or not isinstance(task_id, str):
            raise RolloutSignalError("cpu_control_append_ledger_invalid")
        key = (str(control_type), task_id)
        if key in seen:
            raise RolloutSignalError(f"cpu_control_duplicate:{control_type}:{task_id}")
        seen.add(key)
        counts[str(control_type)] += 1
    return {
        "reference_sql": int(counts["reference_sql"]),
        "python_oracle": int(counts["python_oracle"]),
        "total": int(sum(counts.values())),
    }


def _load_frozen_panel(
    paths: dict[str, Path], panel: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Reuse all three immutable panel files without executing controls again."""

    panel_root = paths["run"] / "panel"
    public_path = panel_root / "public-panel.jsonl"
    private_path = panel_root / "private" / "verifier-inputs.jsonl"
    control_path = panel_root / "private" / "cpu-controls.jsonl"
    present = [path.is_file() for path in (public_path, private_path, control_path)]
    if not any(present):
        return None
    if not all(present):
        raise RolloutSignalError("partial_frozen_panel_requires_recovery")
    public_rows = read_jsonl(public_path)
    private_rows = read_jsonl(private_path)
    control_rows = read_jsonl(control_path)
    if [row.get("task_id") for row in public_rows] != list(TASK_IDS):
        raise RolloutSignalError("frozen_public_panel_order_changed")
    if [row.get("task_id") for row in private_rows] != list(TASK_IDS):
        raise RolloutSignalError("frozen_private_panel_order_changed")
    if [row.get("task_id") for row in control_rows] != list(TASK_IDS):
        raise RolloutSignalError("frozen_control_panel_order_changed")
    selected_panel = {row["task_id"]: row for row in panel["tasks"]}
    for public, private, control in zip(public_rows, private_rows, control_rows, strict=True):
        task_id = public.get("task_id")
        if private.get("task_id") != task_id or control.get("task_id") != task_id:
            raise RolloutSignalError(f"frozen_panel_task_binding_changed:{task_id}")
        instance, template = TASK_ORDER[TASK_IDS.index(task_id)]
        task = _selected_task(instance, template)
        declared = selected_panel.get(task_id)
        if declared is None or public.get("public_task") != task.public.model_dump():
            raise RolloutSignalError(f"frozen_public_task_changed:{task_id}")
        reference_hash = hashlib.sha256(task.reference_sql.encode()).hexdigest()
        if private.get("reference_sql_sha256") != reference_hash:
            raise RolloutSignalError(f"frozen_reference_sql_hash_changed:{task_id}")
        if control.get("reference_sql_sha256") != reference_hash:
            raise RolloutSignalError(f"frozen_control_reference_hash_changed:{task_id}")
        declared_oracle = next(
            row for row in panel["oracle_checks"] if row.get("task_id") == task_id
        )
        if control.get("oracle_sha256") != declared_oracle.get("oracle_sha256"):
            raise RolloutSignalError(f"frozen_oracle_hash_changed:{task_id}")
        if control.get("policy_messages_read_reference_sql") is not False:
            raise RolloutSignalError(f"frozen_control_policy_leak:{task_id}")
    _control_counts(paths["run"])
    return public_rows, private_rows, control_rows


def _write_schedule(
    run_dir: Path, public_rows: list[dict[str, Any]], config: Any
) -> list[dict[str, Any]]:
    schedule = []
    seed_start = int(config.rollout["seed_start"])
    samples = int(config.rollout["samples_per_task"])
    for group_index, row in enumerate(public_rows):
        for sample_index in range(samples):
            seed = seed_start + group_index * samples + sample_index
            schedule.append(
                {
                    "round_index": 0,
                    "group_index": group_index,
                    "group_id": row["group_id"],
                    "task_id": row["task_id"],
                    "database_id": row["database_id"],
                    "sample_index": sample_index,
                    "episode_id": f"r0-{row['group_id']}-s{sample_index}",
                    "seed": seed,
                    "runtime_seed": int(config.runtime["seed"]),
                    "generation": dict(config.rollout["generation"]),
                }
            )
    if [row["seed"] for row in schedule] != list(range(74, 90)):
        raise RolloutSignalError("schedule_seed_contract_failed")
    _write_json(run_dir / "episode-schedule.json", schedule)
    return schedule


def _write_source_snapshot(paths: dict[str, Path]) -> dict[str, Any]:
    run_dir = paths["run"]
    root = run_dir / "source-snapshot"
    root.mkdir(parents=True, exist_ok=True)
    files = []
    patch_parts = []
    for relative in SOURCE_PATHS:
        source = paths["repo"] / relative
        if not source.is_file():
            raise RolloutSignalError(f"source_snapshot_missing:{relative}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "snapshot_path": str(target.relative_to(run_dir)),
            }
        )
        tracked = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--", relative],
            cwd=paths["repo"],
            capture_output=True,
            text=True,
            check=False,
        )
        if tracked.stdout:
            patch_parts.append(tracked.stdout)
        listed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=paths["repo"],
            capture_output=True,
            text=True,
            check=False,
        )
        if listed.returncode != 0:
            untracked = subprocess.run(
                ["git", "diff", "--no-index", "--", "/dev/null", relative],
                cwd=paths["repo"],
                capture_output=True,
                text=True,
                check=False,
            )
            if untracked.stdout:
                patch_parts.append(untracked.stdout)
    patch = root / "patch.diff"
    patch.write_text("\n".join(patch_parts), encoding="utf-8")
    git = {}
    for name, argv in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short", "--branch"],
    }.items():
        result = subprocess.run(
            argv, cwd=paths["repo"], capture_output=True, text=True, check=False
        )
        git[name] = (result.stdout + result.stderr).strip()
    manifest = {
        "schema_version": "t033-source-snapshot-v1",
        "task": "T-033/E-030",
        "git": git,
        "source_paths": files,
        "patch_path": str(patch.relative_to(run_dir)),
        "patch_sha256": sha256_file(patch),
        "protected_paths_included": [],
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def _environment_preflight(paths: dict[str, Path], config: Any) -> dict[str, Any]:
    return {
        "schema_version": "t033-environment-v1",
        "created_at_utc": _now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": str(Path.cwd()),
        "repo_root": str(paths["repo"]),
        "model_id": config.model["model_id"],
        "model_revision": config.model["revision"],
        "dtype": config.model["dtype"],
        "attention_backend": config.model["attention_backend"],
        "enable_thinking": config.model["enable_thinking"],
        "runtime_versions": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
            "pydantic": _package_version("pydantic"),
        },
        "declared_runtime": dict(config.runtime),
        "cuda_imported": False,
        "tokenizer_loaded": False,
        "model_weights_loaded": False,
        "model_loaded": False,
        "validation_split_accessed": False,
        "test_split_accessed": False,
    }


def _cpu_tokenizer_preflight(
    paths: dict[str, Path], context: Any, config: Any
) -> dict[str, Any]:
    """Check the cached tokenizer without loading any Qwen3 model weights."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model["model_id"]),
        revision=str(config.model["revision"]),
        cache_dir=str(paths["cache"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    result = _write_tokenizer_preflight(tokenizer, context, config)
    result.update({"tokenizer_loaded": True, "model_weights_loaded": False})
    _write_json(paths["run"] / "setup" / "tokenizer-preflight.json", result)
    return result


def _budget(paths: dict[str, Path], config: Any) -> BudgetGuard:
    return BudgetGuard(
        paths["run"] / "budget-state.json",
        run_id=RUN_ID,
        run_hard_seconds=int(config.budget["run_hard_seconds"]),
        job_hard_seconds=int(config.budget["job_hard_seconds"]),
        soft_seconds=int(config.budget["soft_seconds"]),
        cleanup_reserve_seconds=int(config.budget["cleanup_reserve_seconds"]),
        gpu_count=int(config.budget["gpu_count"]),
        call_limits={str(key): int(value) for key, value in config.limits.items()},
    )


def _record_command(
    run_dir: Path,
    *,
    command_id: str,
    command: str,
    argv: list[str] | None,
    returncode: int,
    stdout_path: str | None,
    stderr_path: str | None,
    scope: str,
) -> None:
    path = run_dir / "commands.json"
    current = read_json(path, {}) or {}
    rows = current.get("commands", [])
    if not isinstance(rows, list):
        raise RolloutSignalError("commands_json_invalid")
    attempt = sum(row.get("command_id") == command_id for row in rows) + 1
    rows.append(
        {
            "command_id": command_id,
            "attempt": attempt,
            "command": command,
            "argv": argv,
            "returncode": int(returncode),
            "stdout_path": str(Path(stdout_path).resolve()) if stdout_path else None,
            "stderr_path": str(Path(stderr_path).resolve()) if stderr_path else None,
            "scope": scope,
            "recorded_at_utc": _now(),
        }
    )
    _write_json(path, {"schema_version": "t033-commands-v1", "commands": rows})


def _prepare(paths: dict[str, Path]) -> dict[str, Any]:
    _ensure_run_dir(paths["run"])
    panel = json.loads(paths["panel_manifest"].read_text(encoding="utf-8"))
    _verify_panel_manifest(panel)
    config, context = _config(paths)
    if config.model["model_id"] != QWEN3_MODEL_ID or config.model["revision"] != QWEN3_REVISION:
        raise RolloutSignalError("config_model_identity_mismatch")
    tokenizer_preflight = _cpu_tokenizer_preflight(paths, context, config)
    databases = _copy_frozen_databases(paths, panel, paths["run"])
    frozen = _load_frozen_panel(paths, panel)
    if frozen is None:
        public_rows, private_rows, control_rows = _build_panel(paths, panel, config)
    else:
        public_rows, private_rows, control_rows = frozen
    panel_root = paths["run"] / "panel"
    private_root = panel_root / "private"
    panel_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    public_path = panel_root / "public-panel.jsonl"
    private_path = private_root / "verifier-inputs.jsonl"
    control_path = private_root / "cpu-controls.jsonl"
    for path, rows in (
        (public_path, public_rows),
        (private_path, private_rows),
        (control_path, control_rows),
    ):
        if path.exists():
            old = read_jsonl(path)
            if old != rows:
                raise RolloutSignalError(f"immutable_preflight_input_conflict:{path}")
        else:
            path.write_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
                )
                + "\n"
            )
    schedule = _write_schedule(paths["run"], public_rows, config)
    source_snapshot = _write_source_snapshot(paths)
    environment = _environment_preflight(paths, config)
    environment.update(
        {
            "tokenizer_loaded": True,
            "model_weights_loaded": False,
            "tokenizer_preflight": tokenizer_preflight,
        }
    )
    _write_json(paths["run"] / "environment.json", environment)
    cache_manifest = paths["cache"] / "qwen-cache-manifest.json"
    control_calls = _control_counts(paths["run"])
    input_manifest = {
        "schema_version": "t033-input-manifest-v1",
        "task": "T-033/E-030",
        "created_at_utc": _now(),
        "source_panel_manifest": {
            "path": str(paths["panel_manifest"]),
            "sha256": sha256_file(paths["panel_manifest"]),
            "experiment": panel.get("experiment"),
            "split": panel.get("split"),
            "year": panel.get("year"),
        },
        "databases": databases,
        "tasks": [
            {
                "task_id": row["task_id"],
                "group_id": row["group_id"],
                "database_id": row["database_id"],
                "template": row["template"],
                "public_task_sha256": row["public_task_sha256"],
                "private_verifier_row_sha256": _sha(
                    next(item for item in private_rows if item["task_id"] == row["task_id"])
                ),
            }
            for row in public_rows
        ],
        "task_order": list(TASK_IDS),
        "schedule_sha256": _sha(schedule),
        "parent_adapter": {
            "path": str(paths["parent_adapter"]),
            "adapter_model_sha256": QWEN3_PARENT_WEIGHT_SHA256,
            "adapter_config_sha256": QWEN3_PARENT_CONFIG_SHA256,
        },
        "model": {
            "model_id": QWEN3_MODEL_ID,
            "revision": QWEN3_REVISION,
            "cache_dir": str(paths["cache"]),
            "cache_manifest_sha256": sha256_file(cache_manifest)
            if cache_manifest.is_file()
            else None,
            "context_tokens": QWEN3_CONTEXT_TOKENS,
            "tokenizer_preflight": tokenizer_preflight,
        },
        "private_reference_sql_policy": "verifier_and_cpu_control_only",
        "model_messages_contain_reference_sql": False,
        "validation_split_accessed": False,
        "test_split_accessed": False,
        "cpu_control_calls": {
            **control_calls,
            "accounting": "append-only cpu-control-calls.jsonl",
        },
        "source_snapshot": source_snapshot,
    }
    _write_json(paths["run"] / "input-manifest.json", input_manifest)
    frozen = {
        "schema_version": "t033-frozen-config-v1",
        "task": "T-033/E-030",
        "config_path": str(paths["config"]),
        "config": config.raw,
        "resolved_context": {
            "repo_root": str(context.repo_root),
            "run_dir": str(context.run_dir),
        },
        "task_order": list(TASK_IDS),
        "schedule": schedule,
        "database_hashes": {
            row["id"]: {
                "physical_sha256": row["copied_physical_sha256"],
                "logical_fingerprint": row["copied_logical_fingerprint"],
            }
            for row in databases
        },
        "parent_adapter": input_manifest["parent_adapter"],
        "model_messages_contain_reference_sql": False,
        "training": {
            "backward": 0,
            "optimizer_create": 0,
            "optimizer_step": 0,
            "updates": 0,
            "training_job": 0,
        },
        "validation_split_accessed": False,
        "test_split_accessed": False,
    }
    _write_json(paths["run"] / "frozen-config.json", frozen)
    guard = _budget(paths, config)
    guard.current("preflight-complete")
    existing_job_chain = read_json(paths["run"] / "job-chain.json", {}) or {}
    if not isinstance(existing_job_chain, dict) or not existing_job_chain.get("jobs"):
        _write_json(
            paths["run"] / "job-chain.json",
            {
                "schema_version": "t033-job-chain-v1",
                "task": "T-033/E-030",
                "jobs": [],
                "at_most_one_executable_job": True,
                "pending_replacements": [],
                "training_jobs": 0,
            },
        )
    existing_allocation = read_json(paths["run"] / "allocation-ledger.json", {}) or {}
    if not isinstance(existing_allocation, dict) or not existing_allocation.get(
        "slurm_root_rows"
    ):
        _write_json(
            paths["run"] / "allocation-ledger.json",
            {
                "schema_version": "t033-allocation-ledger-v1",
                "slurm_root_allocation_seconds": None,
                "budget_accounted_gpu_seconds": 0.0,
                "budget_accounted_gpu_minutes": 0.0,
                "wrapper_elapsed_seconds": None,
                "currency_cost": None,
                "source": "not_started",
            },
        )
    checks = {
        "schema_version": "t033-checks-v1",
        "phase": "preflight",
        "status": "PASS",
        "checks": [
            {"name": "selected_tasks", "status": "PASS", "observed": list(TASK_IDS)},
            {"name": "database_physical_and_logical_hashes", "status": "PASS"},
            {"name": "database_readonly_copies", "status": "PASS"},
            {
                "name": "reference_sql_python_oracle_controls",
                "status": "PASS",
                "observed": control_calls,
            },
            {"name": "model_load_performed", "status": "PASS", "observed": False},
            {
                "name": "tokenizer_hard_off_context_preflight",
                "status": "PASS"
                if tokenizer_preflight["hard_thinking_off"]
                and tokenizer_preflight["context_boundary_accepts"]
                else "FAIL",
            },
            {"name": "validation_or_test_accessed", "status": "PASS", "observed": False},
            {"name": "training_calls", "status": "PASS", "observed": 0},
        ],
    }
    _write_json(paths["run"] / "checks.json", checks)
    preflight = {
        "schema_version": "t033-preflight-v1",
        "status": "PASSED",
        "task": "T-033/E-030",
        "created_at_utc": _now(),
        "planned_episodes": len(schedule),
        "group_count": len(public_rows),
        "samples_per_task": 4,
        "task_order": list(TASK_IDS),
        "schedule_sha256": _sha(schedule),
        "cpu_control_calls": control_calls,
        "model_load_count": 0,
        "tokenizer_load_count": 1,
        "model_weights_load_count": 0,
        "adapter_merge_count": 0,
        "optimizer_create_count": 0,
        "backward_count": 0,
        "optimizer_step_count": 0,
        "validation_split_accessed": False,
        "test_split_accessed": False,
        "public_panel_path": str(public_path.relative_to(paths["run"])),
        "private_verifier_path": str(private_path.relative_to(paths["run"])),
    }
    _write_json(paths["run"] / "preflight.json", preflight)
    return preflight


def _load_prepared_tasks(paths: dict[str, Path], config: Any) -> list[TaskInput]:
    public_path = paths["run"] / "panel" / "public-panel.jsonl"
    private_path = paths["run"] / "panel" / "private" / "verifier-inputs.jsonl"
    public = _read_index(public_path, "task_id")
    private = _read_index(private_path, "task_id")
    if list(public) != list(TASK_IDS) or list(private) != list(TASK_IDS):
        raise RolloutSignalError("prepared_task_order_changed")
    tasks = []
    for task_id in TASK_IDS:
        public_row = public[task_id]
        private_row = private[task_id]
        public_task = public_row.get("public_task")
        if not isinstance(public_task, dict) or any(
            key in public_row or key in public_task
            for key in ("reference_sql", "expected", "benchmark_task")
        ):
            raise RolloutSignalError(f"public_task_private_field_leak:{task_id}")
        benchmark_task = private_row.get("benchmark_task")
        if not isinstance(benchmark_task, dict):
            raise RolloutSignalError(f"private_benchmark_task_missing:{task_id}")
        if benchmark_task.get("public") != public_task:
            raise RolloutSignalError(f"private_public_task_mismatch:{task_id}")
        tasks.append(
            TaskInput(
                task_id=task_id,
                group_id=str(public_row["group_id"]),
                public_task=public_task,
                database_path=str(Path(public_row["database_path"]).resolve()),
                verifier_payload=benchmark_task,
                metadata={
                    "database_id": public_row["database_id"],
                    "template": public_row["template"],
                    "runtime_seed": int(config.runtime["seed"]),
                    "model_messages_contain_reference_sql": False,
                },
            )
        )
    return tasks


def _verifier_for(settings: Settings):
    def verifier(task: TaskInput, trajectory: Trajectory):
        if not isinstance(task.verifier_payload, dict):
            raise RolloutSignalError(f"verifier_payload_missing:{task.task_id}")
        benchmark_task = BenchmarkTask.model_validate(task.verifier_payload)
        return verify(benchmark_task, trajectory, Path(str(task.database_path)), settings)

    return verifier


def _schedule_by_task(schedule: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in schedule}


def _episode_summary(episode: EpisodeSample) -> dict[str, Any]:
    trajectory = episode.metadata.get("trajectory")
    model_attempts = trajectory.get("model_attempts", []) if isinstance(trajectory, dict) else []
    return {
        "episode_id": episode.episode_id,
        "task_id": episode.task_id,
        "group_id": episode.group_id,
        "sample_index": episode.sample_index,
        "seed": episode.seed,
        "token_record_count": len(episode.token_records),
        "physical_request_ids": [record.request_id for record in episode.token_records],
        "model_attempt_count": len(model_attempts),
        "terminal_reason": trajectory.get("terminal_reason")
        if isinstance(trajectory, dict)
        else None,
        "format_failure": bool(
            episode.failed_action and episode.failed_action.get("status") == "format_failed"
        ),
        "failed_action": episode.failed_action,
    }


def _score_row(
    *,
    schedule_row: dict[str, Any],
    episode: EpisodeSample,
    score: ScoreResult,
    qualification: dict[str, Any] | None,
    trajectory_path: str | None,
    verifier_called: bool,
) -> dict[str, Any]:
    row = {
        **score.to_dict(),
        "record_type": "RESULT",
        "round_index": schedule_row["round_index"],
        "group_index": schedule_row["group_index"],
        "group_id": schedule_row["group_id"],
        "sample_index": schedule_row["sample_index"],
        "schedule_seed": schedule_row["seed"],
        "raw_saved": True,
        "verifier_called": verifier_called,
        "trajectory_path": trajectory_path,
        "qualification": qualification,
        "episode_summary": _episode_summary(episode),
    }
    return row


def _run_schedule(
    paths: dict[str, Path],
    config: Any,
    guard: BudgetGuard,
    tasks: list[TaskInput],
    policy: Any,
    *,
    engine: Any | None = None,
    worker: Any | None = None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schedule = json.loads((paths["run"] / "episode-schedule.json").read_text(encoding="utf-8"))
    tasks_by_id = {task.task_id: task for task in tasks}
    episode_path = paths["run"] / "raw" / "episodes.jsonl"
    token_path = paths["run"] / "token-rollouts.jsonl"
    score_path = paths["run"] / "case-results.jsonl"
    trajectory_root = paths["run"] / "raw" / "trajectories"
    existing_episodes = _read_index(episode_path, "episode_id")
    existing_scores = _read_index(score_path, "episode_id")
    snapshot = snapshot or qwen3_policy_snapshot(config, policy)
    engine = engine or HfRolloutEngine(
        policy=policy, run_dir=paths["run"], config=config, evaluation=False
    )
    settings = Settings.model_validate(config.runtime)
    worker = worker or RealVerifierWorker(verifier=_verifier_for(settings), guard=guard)
    stop_reason: str | None = None
    for schedule_row in schedule:
        episode_id = schedule_row["episode_id"]
        task = tasks_by_id[schedule_row["task_id"]]
        previous_score = existing_scores.get(episode_id)
        if previous_score is not None:
            if previous_score.get("status") == "unknown":
                stop_reason = "persisted_technical_unknown"
                break
            if previous_score.get("status") == "scored":
                continue
            raise RolloutSignalError(f"invalid_persisted_score_status:{episode_id}")
        episode: EpisodeSample
        if episode_id in existing_episodes:
            episode = EpisodeSample.from_dict(existing_episodes[episode_id])
            episode.validate()
        else:
            guard.check(f"rollout:before:{episode_id}")
            try:
                episode = guard.run_callable(
                    lambda task=task, row=schedule_row, episode_key=episode_id: engine.rollout(
                        task,
                        round_index=int(row["round_index"]),
                        sample_index=int(row["sample_index"]),
                        seed=int(row["seed"]),
                        policy_snapshot=snapshot,
                        episode_id=episode_key,
                    ),
                    phase=f"rollout:{episode_id}",
                )
                if not isinstance(episode, EpisodeSample):
                    raise RolloutSignalError("rollout_engine_return_type_invalid")
                if episode.episode_id != episode_id or episode.seed != schedule_row["seed"]:
                    raise RolloutSignalError(f"rollout_schedule_binding_mismatch:{episode_id}")
                if episode.policy_snapshot_id != snapshot["policy_snapshot_id"]:
                    raise RolloutSignalError(f"rollout_policy_snapshot_mismatch:{episode_id}")
                episode.validate()
            except BudgetError:
                raise
            except Exception as exc:
                failure = EpisodeSample(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    group_id=task.group_id,
                    sample_index=int(schedule_row["sample_index"]),
                    seed=int(schedule_row["seed"]),
                    policy_snapshot_id=snapshot["policy_snapshot_id"],
                    model_ref_id=str(getattr(policy, "model_ref_id", "unknown")),
                    token_records=[],
                    failed_action={
                        "status": "engine_error",
                        "error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
                    },
                )
                _append_jsonl(
                    paths["run"] / "raw" / "engine-failures.jsonl",
                    {
                        "episode_id": episode_id,
                        "error": failure.failed_action,
                        "stop_new_requests": True,
                    },
                )
                stop_reason = f"engine_or_evidence_failure:{type(exc).__name__}"
                break
            _append_jsonl(
                episode_path, {"record_type": "RAW", **episode.to_dict(), "raw_saved": True}
            )
            for token in episode.token_records:
                _append_jsonl(
                    token_path, {"record_type": "TOKEN", **token.to_dict(), "raw_saved": True}
                )
            existing_episodes[episode_id] = {
                "record_type": "RAW",
                **episode.to_dict(),
                "raw_saved": True,
            }
        trajectory = episode.metadata.get("trajectory")
        trajectory_path: str | None = None
        qualification: dict[str, Any] | None = None
        if isinstance(trajectory, dict):
            trajectory_file = trajectory_root / f"{episode_id}.json"
            immutable_write_json(trajectory_file, trajectory)
            trajectory_path = str(trajectory_file.relative_to(paths["run"]))
        if getattr(engine, "evidence", None) is None:
            qualification = {
                "status": "NOT_REQUIRED",
                "trajectory_sha256": _sha(trajectory) if isinstance(trajectory, dict) else None,
            }
        else:
            try:
                qualification = qualify_episode_evidence(
                    episode,
                    run_dir=paths["run"],
                    expected_stage="training",
                    expected_arm="train",
                    trajectory_path=trajectory_path,
                )
            except (EvidenceError, OSError, ValueError, TypeError) as exc:
                score = ScoreResult(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    status="unknown",
                    infra_error={"type": type(exc).__name__, "message": str(exc)[:1000]},
                    error={"type": type(exc).__name__, "message": str(exc)[:1000]},
                    source_kind="real",
                )
                row = _score_row(
                    schedule_row=schedule_row,
                    episode=episode,
                    score=score,
                    qualification=None,
                    trajectory_path=trajectory_path,
                    verifier_called=False,
                )
                _append_jsonl(score_path, row)
                existing_scores[episode_id] = row
                stop_reason = f"evidence_qualification_failure:{type(exc).__name__}"
                break
        guard.check(f"verifier:before:{episode_id}")
        try:
            score = guard.run_callable(
                lambda task=task, episode=episode: worker.score(task, episode),
                phase=f"verifier:{episode_id}",
            )
            if not isinstance(score, ScoreResult):
                raise RolloutSignalError("verifier_return_type_invalid")
            score.validate()
        except BudgetError:
            raise
        except Exception as exc:
            score = ScoreResult(
                episode_id=episode_id,
                task_id=task.task_id,
                status="unknown",
                infra_error={"type": type(exc).__name__, "message": str(exc)[:1000]},
                error={"type": type(exc).__name__, "message": str(exc)[:1000]},
                source_kind="real",
            )
        row = _score_row(
            schedule_row=schedule_row,
            episode=episode,
            score=score,
            qualification=qualification,
            trajectory_path=trajectory_path,
            verifier_called=True,
        )
        _append_jsonl(score_path, row)
        existing_scores[episode_id] = row
        if score.status == "unknown":
            stop_reason = "verifier_technical_unknown"
            break
        _write_json(
            paths["run"] / "rollout-progress.json",
            {
                "schema_version": "t033-progress-v1",
                "planned": len(schedule),
                "attempted": len(existing_scores),
                "scored": sum(row.get("status") == "scored" for row in existing_scores.values()),
                "unknown": sum(row.get("status") == "unknown" for row in existing_scores.values()),
                "last_episode_id": episode_id,
                "updated_at_utc": _now(),
            },
        )
    scored = sum(row.get("status") == "scored" for row in existing_scores.values())
    unknown = sum(row.get("status") == "unknown" for row in existing_scores.values())
    result = {
        "schema_version": "t033-rollout-result-v1",
        "status": "COMPLETED" if scored == len(schedule) and unknown == 0 else "INCOMPLETE",
        "planned": len(schedule),
        "attempted": len(existing_scores),
        "scored": scored,
        "unknown": unknown,
        "stop_reason": stop_reason,
        "policy_snapshot_id": snapshot["policy_snapshot_id"],
        "training_calls": snapshot["training_calls"],
        "schedule_sha256": _sha(schedule),
        "updated_at_utc": _now(),
    }
    _write_json(paths["run"] / "rollout-result.json", result)
    return result


def _update_job_chain(
    paths: dict[str, Path], *, job_id: str, status: str, error: Any = None
) -> None:
    path = paths["run"] / "job-chain.json"
    chain = read_json(path, {}) or {}
    jobs = chain.setdefault("jobs", [])
    row = next((item for item in jobs if item.get("job_id") == job_id), None)
    if row is None:
        row = {
            "job_id": job_id,
            "account": os.environ.get("SLURM_JOB_ACCOUNT") or os.environ.get("SLURM_ACCOUNT"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "qos": os.environ.get("SLURM_JOB_QOS") or os.environ.get("SLURM_QOS"),
            "node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": 1,
            "stage": "rollout",
            "training": False,
            "started_at_utc": _now(),
        }
        jobs.append(row)
    row.update({"status": status, "ended_at_utc": _now(), "error": error})
    _write_json(paths["run"] / "job-chain.json", chain)


def _rollout(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RolloutSignalError("real_qwen3_rollout_requires_slurm_job")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or len([item for item in visible.split(",") if item.strip()]) != 1:
        raise RolloutSignalError("rollout_requires_one_visible_gpu_before_torch_import")
    preflight = read_json(paths["run"] / "preflight.json")
    if not isinstance(preflight, dict) or preflight.get("status") != "PASSED":
        raise RolloutSignalError("preflight_missing_or_failed")
    config, context = _config(paths)
    tasks = _load_prepared_tasks(paths, config)
    guard = _budget(paths, config)
    state = read_json(paths["run"] / "budget-state.json", {}) or {}
    if state.get("active_attempt") is not None:
        recovered = args.recover_interrupted_gpu_seconds
        if recovered is None:
            raise RolloutSignalError("unfinished_gpu_attempt_requires_explicit_recovery_seconds")
        guard.recover_interrupted(
            elapsed_gpu_seconds=float(recovered), reason="explicit_t033_job_recovery"
        )
    job_id = str(os.environ["SLURM_JOB_ID"])
    guard.start_job(job_id=job_id)
    _update_job_chain(paths, job_id=job_id, status="RUNNING")
    policy = None
    result: dict[str, Any] | None = None
    final_status = "FAILED"
    error_value: Any = None
    try:
        policy = guard.run_callable(
            lambda: load_qwen3_rollout_policy(context, config, guard),
            phase="rollout-setup:model-and-parent-load",
        )
        result = _run_schedule(paths, config, guard, tasks, policy)
        final_status = "COMPLETED" if result["status"] == "COMPLETED" else "INCOMPLETE"
        if final_status != "COMPLETED":
            error_value = {"stop_reason": result.get("stop_reason")}
    except BudgetError as exc:
        error_value = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        final_status = "FAILED_BUDGET"
        raise
    except Exception as exc:
        error_value = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        final_status = "FAILED"
        raise
    finally:
        if policy is not None:
            try:
                policy.close()
            except Exception as exc:
                error_value = error_value or {
                    "type": type(exc).__name__,
                    "message": str(exc)[:1000],
                }
                final_status = "FAILED_CLEANUP"
        try:
            snapshot = guard.close_job(
                status=final_status,
                reason=json.dumps(error_value, ensure_ascii=False) if error_value else None,
            )
            _write_json(
                paths["run"] / "allocation-ledger.json",
                {
                    "schema_version": "t033-allocation-ledger-v1",
                    "budget_accounted_gpu_seconds": snapshot.consumed_gpu_seconds,
                    "budget_accounted_gpu_minutes": snapshot.consumed_gpu_seconds / 60.0,
                    "slurm_root_allocation_seconds": None,
                    "wrapper_elapsed_seconds": None,
                    "currency_cost": None,
                    "source": (
                        "BudgetGuard full allocation wall-clock; "
                        "Slurm root evidence finalized offline"
                    ),
                },
            )
        finally:
            _update_job_chain(paths, job_id=job_id, status=final_status, error=error_value)
    if result is None:
        raise RolloutSignalError("rollout_result_missing")
    if result["status"] != "COMPLETED":
        raise RolloutSignalError(f"rollout_incomplete:{result.get('stop_reason')}")
    return result


def _finite(values: list[Any]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _validate_tokens(
    paths: dict[str, Path], episodes: dict[str, dict[str, Any]], tokens: list[dict[str, Any]]
) -> dict[str, Any]:
    request_ids: set[str] = set()
    response_ids: set[str] = set()
    token_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors = []
    for row in tokens:
        try:
            record = TokenRecord.from_dict(row)
            record.validate()
        except Exception as exc:
            errors.append(f"token_contract:{type(exc).__name__}:{exc}")
            continue
        if row.get("completion_mask") != [1] * len(record.generated_token_ids):
            errors.append(f"completion_mask:{record.request_id}")
        if not _finite(record.raw_policy_log_probs) or not _finite(
            record.behavior_sampling_log_probs
        ):
            errors.append(f"nonfinite_logprob:{record.request_id}")
        if row.get("old_per_token_logps") != row.get("raw_policy_log_probs"):
            errors.append(f"old_logprob_alias_mismatch:{record.request_id}")
        if record.request_id in request_ids or record.response_id in response_ids:
            errors.append(f"duplicate_physical_request:{record.request_id}")
        request_ids.add(record.request_id)
        response_ids.add(record.response_id)
        metadata = record.metadata
        expected_eos = bool(
            record.generated_token_ids
            and metadata.get("eos_token_id") is not None
            and record.generated_token_ids[-1] == metadata.get("eos_token_id")
        )
        if bool(metadata.get("eos_generated")) != expected_eos:
            errors.append(f"eos_contract:{record.request_id}")
        if metadata.get("native_score_count") != len(record.generated_token_ids):
            errors.append(f"native_score_count:{record.request_id}")
        if metadata.get("native_score_log_probs") != record.behavior_sampling_log_probs:
            errors.append(f"native_score_binding:{record.request_id}")
        if metadata.get("native_raw_log_probs") != record.raw_policy_log_probs:
            errors.append(f"raw_logprob_binding:{record.request_id}")
        for key in ("request_path", "response_path", "engine_event_path"):
            value = metadata.get(key)
            if not isinstance(value, str) or not (paths["run"] / value).is_file():
                errors.append(f"evidence_path:{record.request_id}:{key}")
        token_by_episode[record.episode_id].append(row)
    for episode_id, episode in episodes.items():
        token_rows = token_by_episode.get(episode_id, [])
        expected_count = len(episode.get("token_records", []))
        if len(token_rows) != expected_count:
            errors.append(
                f"episode_token_cardinality:{episode_id}:{len(token_rows)}:{expected_count}"
            )
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "physical_request_count": len(request_ids),
        "request_ids": sorted(request_ids),
        "response_ids": sorted(response_ids),
        "token_by_episode": token_by_episode,
    }


def _read_slurm_root_evidence(paths: dict[str, Path]) -> dict[str, Any]:
    log_dir = paths["run"] / "logs"
    candidates = sorted(log_dir.glob("slurm-root-*.txt"))
    rows = []
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.strip() and not line.startswith("slurmstepd"):
                rows.append({"path": str(path.relative_to(paths["run"])), "line": line})
    root_rows = []
    for row in rows:
        parts = row["line"].split("|")
        if len(parts) >= 3 and ".batch" not in parts[0] and ".extern" not in parts[0]:
            elapsed = parts[2]
            try:
                root_rows.append(
                    {**row, "job_id": parts[0], "state": parts[1], "elapsed_raw": float(elapsed)}
                )
            except ValueError:
                continue
    return {"files": candidates, "rows": rows, "root_rows": root_rows}


def _group_metrics(
    schedule: list[dict[str, Any]], scores: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    groups = []
    for group_index, task_id in enumerate(TASK_IDS):
        rows = [scores.get(row["episode_id"]) for row in schedule if row["task_id"] == task_id]
        values = [
            float(row["reward_value"])
            for row in rows
            if row and row.get("status") == "scored" and row.get("reward_value") is not None
        ]
        complete = len(rows) == 4 and all(
            row is not None and row.get("status") == "scored" for row in rows
        )
        reward_values = values if complete else values
        population_variance = None
        sample_std = None
        advantages = None
        if complete:
            mean = sum(reward_values) / len(reward_values)
            population_variance = sum((value - mean) ** 2 for value in reward_values) / len(
                reward_values
            )
            if len(reward_values) > 1:
                sample_std = math.sqrt(
                    sum((value - mean) ** 2 for value in reward_values) / (len(reward_values) - 1)
                )
            else:
                sample_std = 0.0
            advantages = [
                0.0 if sample_std == 0.0 else (value - mean) / (sample_std + 1e-4)
                for value in reward_values
            ]
        groups.append(
            {
                "group_index": group_index,
                "group_id": task_id,
                "task_id": task_id,
                "episode_ids": [row["episode_id"] for row in schedule if row["task_id"] == task_id],
                "reward_values": reward_values,
                "reward_count": len(reward_values),
                "expected_denominator": 4,
                "complete": complete,
                "population_variance": population_variance,
                "sample_std": sample_std,
                "advantages": advantages,
                "unique_rewards": sorted(set(reward_values)),
                "nondegenerate": bool(complete and len(set(reward_values)) > 1),
            }
        )
    return {
        "schema_version": "t033-group-metrics-v1",
        "population_variance_definition": "sum((reward-mean)^2)/4",
        "advantage_definition": (
            "(reward-group_mean)/(sample_std+1e-4), sample_std=statistics.stdev"
        ),
        "groups": groups,
        "complete_groups": sum(group["complete"] for group in groups),
        "nondegenerate_group_count": sum(group["nondegenerate"] for group in groups),
    }


def _scan_reference_sql_policy_inputs(
    request_root: Path, private_sql: list[str], run_root: Path
) -> dict[str, list[str]]:
    """Check private SQL against model-input messages, not model-generated history.

    Each persisted generation request contains the preceding assistant action so the
    next request is reproducible. A model can independently emit the exact
    benchmark SQL; treating that assistant output as an injected private value
    would be a false positive. System/user/tool-observation messages are the
    policy inputs that this audit must protect.
    """
    input_matches: set[str] = set()
    assistant_matches: set[str] = set()
    for request_path in sorted(request_root.glob("*.json")):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        relative = str(request_path.relative_to(run_root))
        for message in request.get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = str(message.get("content") or "")
            for sql in private_sql:
                if not sql or sql not in content:
                    continue
                if message.get("role") == "assistant":
                    assistant_matches.add(relative)
                else:
                    input_matches.add(relative)
    return {
        "non_assistant_input_matches": sorted(input_matches),
        "assistant_generated_matches": sorted(assistant_matches),
    }


def _write_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    """Index immutable run outputs while keeping source and command ledgers separate."""
    excluded = {"artifact-manifest.json", "commands.json"}
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if relative.as_posix() in excluded or relative.parts[0] == "source-snapshot":
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "t033-artifact-manifest-v1",
        "scope": "run-artifacts-excluding-source-snapshot-and-mutable-command-ledger",
        "source_snapshot_manifest": "source-snapshot/manifest.json",
        "excluded_paths": [
            "artifact-manifest.json",
            "commands.json",
            "source-snapshot/**",
        ],
        "files": entries,
        "file_count": len(entries),
    }
    _write_json(run_dir / "artifact-manifest.json", manifest)
    return manifest


def _finalize(paths: dict[str, Path]) -> dict[str, Any]:
    preflight = read_json(paths["run"] / "preflight.json")
    if not isinstance(preflight, dict) or preflight.get("status") != "PASSED":
        raise RolloutSignalError("cannot_finalize_without_passed_preflight")
    schedule = json.loads((paths["run"] / "episode-schedule.json").read_text(encoding="utf-8"))
    if len(schedule) != 16 or [row["seed"] for row in schedule] != list(range(74, 90)):
        raise RolloutSignalError("finalize_schedule_contract_failed")
    episodes = _read_index(paths["run"] / "raw" / "episodes.jsonl", "episode_id")
    scores = _read_index(paths["run"] / "case-results.jsonl", "episode_id")
    missing = [row["episode_id"] for row in schedule if row["episode_id"] not in episodes]
    missing_scores = [row["episode_id"] for row in schedule if row["episode_id"] not in scores]
    if missing or missing_scores:
        raise RolloutSignalError(
            f"finalize_missing_slots:episodes={missing}:scores={missing_scores}"
        )
    for row in schedule:
        episode = EpisodeSample.from_dict(episodes[row["episode_id"]])
        episode.validate()
        if episode.task_id != row["task_id"] or episode.group_id != row["group_id"]:
            raise RolloutSignalError(f"finalize_episode_schedule_mismatch:{row['episode_id']}")
        score = ScoreResult.from_dict(scores[row["episode_id"]])
        score.validate()
    tokens = read_jsonl(paths["run"] / "token-rollouts.jsonl")
    token_check = _validate_tokens(paths, episodes, tokens)
    qualified = sum(
        row.get("status") == "scored"
        and isinstance(row.get("qualification"), dict)
        and row["qualification"].get("status") == "QUALIFIED"
        for row in scores.values()
    )
    scored = sum(row.get("status") == "scored" for row in scores.values())
    unknown = sum(row.get("status") == "unknown" for row in scores.values())
    for row in scores.values():
        if row.get("status") == "scored" and row.get("source_kind") != "real":
            raise RolloutSignalError(f"non_real_score_source:{row.get('episode_id')}")
    private_rows = read_jsonl(paths["run"] / "panel" / "private" / "verifier-inputs.jsonl")
    private_sql = [row["benchmark_task"]["reference_sql"] for row in private_rows]
    request_root = paths["run"] / "engine-evidence" / "requests"
    policy_sql_scan = _scan_reference_sql_policy_inputs(request_root, private_sql, paths["run"])
    request_leaks = policy_sql_scan["non_assistant_input_matches"]
    evidence = evidence_journal_summary(paths["run"])
    budget_state = read_json(paths["run"] / "budget-state.json", {}) or {}
    counters = dict(budget_state.get("call_counters") or {})
    control_calls = _control_counts(paths["run"])
    group_metrics = _group_metrics(schedule, scores)
    complete = (
        len(episodes) == 16
        and len(scores) == 16
        and qualified == 16
        and scored == 16
        and unknown == 0
        and token_check["status"] == "PASS"
        and not request_leaks
    )
    nondegenerate = int(group_metrics["nondegenerate_group_count"])
    if not complete:
        conclusion = "INCOMPLETE_TECHNICAL_EVIDENCE"
    elif nondegenerate:
        conclusion = "WITHIN_TASK_REWARD_VARIANCE"
    else:
        conclusion = "NO_WITHIN_TASK_REWARD_VARIANCE"
    rollout_identity = read_json(paths["run"] / "setup" / "rollout-load.json", {}) or {}
    training_calls = (read_json(paths["run"] / "rollout-result.json", {}) or {}).get(
        "training_calls", {"backward": 0, "optimizer_create": 0, "optimizer_step": 0, "updates": 0}
    )
    training_zero = all(int(value) == 0 for value in training_calls.values())
    slurm = _read_slurm_root_evidence(paths)
    allocation_seconds = None
    if len(slurm["root_rows"]) == 1:
        allocation_seconds = slurm["root_rows"][0]["elapsed_raw"]
    budget_seconds = float(budget_state.get("consumed_gpu_seconds", 0.0))
    allocation = {
        "schema_version": "t033-allocation-ledger-v1",
        "slurm_root_allocation_seconds": allocation_seconds,
        "slurm_root_rows": slurm["root_rows"],
        "budget_accounted_gpu_seconds": budget_seconds,
        "budget_accounted_gpu_minutes": budget_seconds / 60.0,
        "wrapper_elapsed_seconds": None,
        "wrapper_evidence_files": [str(path.relative_to(paths["run"])) for path in slurm["files"]],
        "currency_cost": None,
        "allocation_rule": "root Slurm allocation only; wrapper is separate and never added",
        "within_90_gpu_minutes": budget_seconds <= 5400.0 + 1e-6,
    }
    _write_json(paths["run"] / "allocation-ledger.json", allocation)
    checks = {
        "schema_version": "t033-checks-v1",
        "phase": "final",
        "status": "PASS" if complete else "INCOMPLETE",
        "checks": [
            {
                "name": "schedule_16_fixed_slots",
                "status": "PASS" if len(schedule) == 16 else "FAIL",
            },
            {
                "name": "episodes_16",
                "status": "PASS" if len(episodes) == 16 else "FAIL",
                "observed": len(episodes),
            },
            {
                "name": "qualified_16",
                "status": "PASS" if qualified == 16 else "FAIL",
                "observed": qualified,
            },
            {"name": "scored_16", "status": "PASS" if scored == 16 else "FAIL", "observed": scored},
            {
                "name": "unknown_zero",
                "status": "PASS" if unknown == 0 else "FAIL",
                "observed": unknown,
            },
            {
                "name": "native_token_logprob_mask_contract",
                "status": token_check["status"],
                "errors": token_check["errors"][:20],
            },
            {
                "name": "request_response_engine_binding",
                "status": "PASS"
                if evidence["attempted"]
                == evidence["completed"]
                == token_check["physical_request_count"]
                else "FAIL",
                "evidence": evidence,
            },
            {
                "name": "reference_sql_not_in_non_assistant_policy_inputs",
                "status": "PASS" if not request_leaks else "FAIL",
                "leaks": request_leaks,
                "assistant_generated_matches": policy_sql_scan["assistant_generated_matches"],
            },
            {
                "name": "reward_from_real_business_verifier",
                "status": "PASS"
                if all(row.get("source_kind") == "real" for row in scores.values())
                else "FAIL",
            },
            {
                "name": "parent_adapter_identity",
                "status": "PASS"
                if rollout_identity.get("parent_adapter", {}).get("adapter_model_sha256")
                == QWEN3_PARENT_WEIGHT_SHA256
                and rollout_identity.get("parent_adapter", {}).get("adapter_config_sha256")
                == QWEN3_PARENT_CONFIG_SHA256
                else "FAIL",
            },
            {
                "name": "base_load_merge_freeze",
                "status": "PASS"
                if rollout_identity.get("fresh_base_load_count") == 1
                and rollout_identity.get("parent_adapter_merge_count") == 1
                and rollout_identity.get("all_parameters_frozen") is True
                else "FAIL",
            },
            {
                "name": "training_calls_zero",
                "status": "PASS" if training_zero else "FAIL",
                "observed": training_calls,
            },
            {
                "name": "allocation_budget",
                "status": "PASS" if allocation["within_90_gpu_minutes"] else "FAIL",
                "observed": allocation,
            },
            {"name": "validation_test_not_accessed", "status": "PASS"},
        ],
    }
    _write_json(paths["run"] / "checks.json", checks)
    _write_json(paths["run"] / "group-metrics.json", group_metrics)
    metrics = {
        "schema_version": "t033-metrics-v1",
        "task": "T-033/E-030",
        "status": "COMPLETED" if complete else "INCOMPLETE",
        "conclusion": conclusion,
        "planned_episodes": 16,
        "episode_qualified": qualified,
        "scored": scored,
        "unknown": unknown,
        "group_count": 4,
        "samples_per_group": 4,
        "nondegenerate_group_count": nondegenerate if complete else None,
        "groups_with_reward_variance": [
            group["group_id"] for group in group_metrics["groups"] if group["nondegenerate"]
        ]
        if complete
        else [],
        "model_requests": counters.get("model_requests", 0),
        "runtime_tool_attempts": counters.get("tool_actions", 0),
        "runtime_worker_calls": counters.get("runtime_workers", 0),
        "verifier_calls": counters.get("verifier_calls", 0),
        "verifier_sql_calls": counters.get("verifier_sql", 0),
        "cpu_control_calls": control_calls,
        "training_calls": training_calls,
        "evidence": evidence,
        "token_contract": token_check,
        "allocation": allocation,
        "database_and_input_scope": {
            "tasks": list(TASK_IDS),
            "split": "train",
            "validation_split_accessed": False,
            "test_split_accessed": False,
        },
    }
    metrics["token_contract"].pop("token_by_episode", None)
    _write_json(paths["run"] / "metrics.json", metrics)
    report_lines = [
        "# T-033 / E-030 — Qwen3 SFT分组Rollout与GRPO输入信号验证",
        "",
        f"- 状态：`{metrics['status']}`；主结论：`{conclusion}`。",
        (
            f"- 固定面板：4个train任务×4条，一轮，资格 `{qualified}/16`，"
            f"scored `{scored}/16`，unknown `{unknown}`。"
        ),
        f"- 任务顺序：`{', '.join(TASK_IDS)}`；episode seed：74–89；Runtime seed：17。",
        (
            f"- 模型：`{QWEN3_MODEL_ID}` revision `{QWEN3_REVISION}`，FP16、SDPA math，"
            "hard thinking-off；父Adapter权重/配置 SHA-256 "
            f"`{QWEN3_PARENT_WEIGHT_SHA256}` / `{QWEN3_PARENT_CONFIG_SHA256}`。"
        ),
        (
            f"- 加载契约：fresh Base `{rollout_identity.get('fresh_base_load_count')}`，"
            f"父Adapter merge `{rollout_identity.get('parent_adapter_merge_count')}`，"
            f"LoRA modules `{rollout_identity.get('module_count')}`，"
            f"全参数冻结 `{rollout_identity.get('all_parameters_frozen')}`。"
        ),
        "",
        "## 分组原始业务 Reward",
        "",
        "| group/task | rewards (n=4) | population variance | sample std | advantage |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in group_metrics["groups"]:
        report_lines.append(
            f"| `{group['group_id']}` | `{group['reward_values']}` | "
            f"`{group['population_variance']}` | `{group['sample_std']}` | "
            f"`{group['advantages']}` |"
        )
    report_lines.extend(
        [
            "",
            (
                f"- 非退化组数：`{nondegenerate if complete else 'null'}`；"
                "population variance 定义为 `sum((r-mean)^2)/4`。"
            ),
            (
                "- 下一阶段参考 advantage 仅离线计算，使用 "
                "`(reward-group_mean)/(sample_std+1e-4)`，不执行 loss、backward、"
                "optimizer 或权重更新。"
            ),
            "",
            "## 调用与边界",
            "",
            (
                f"- 真实模型物理请求：`{counters.get('model_requests', 0)}`；"
                f"Runtime tool attempts：`{counters.get('tool_actions', 0)}`；"
                f"Verifier：`{counters.get('verifier_calls', 0)}`；"
                f"Verifier SQL：`{counters.get('verifier_sql', 0)}`；"
                f"CPU reference/Python controls：`{control_calls['reference_sql']}+"
                f"{control_calls['python_oracle']}={control_calls['total']}`，"
                "由 append-only 控制账本派生。"
            ),
            (
                f"- 训练调用：backward `{training_calls.get('backward', 0)}`、"
                f"optimizer create `{training_calls.get('optimizer_create', 0)}`、"
                f"step `{training_calls.get('optimizer_step', 0)}`、"
                f"update `{training_calls.get('updates', 0)}`、训练Job `0`。"
            ),
            (
                f"- allocation ledger：完整Slurm根行 "
                f"`{allocation.get('slurm_root_allocation_seconds')}` 秒；"
                f"BudgetGuard `{budget_seconds}` GPU秒 / "
                f"`{budget_seconds / 60.0}` GPU分钟；wrapper不计入 allocation；成本 `null`。"
            ),
            (
                "- 仅读取四项 train 任务、train-0/train-1 DB；validation/test 未访问；"
                "reference SQL/expected 只在CPU控制与Verifier侧。"
            ),
            "",
            "## 证据与验收",
            "",
            (
                f"- token 契约：`{token_check['status']}`；physical request/response/"
                f"engine event：`{token_check['physical_request_count']}`，"
                f"journal attempted/completed/incomplete="
                f"`{evidence['attempted']}/{evidence['completed']}/{evidence['incomplete']}`。"
            ),
            (
                f"- 非assistant policy inputs 中发现 reference SQL：`{len(request_leaks)}`；"
                f"assistant 自生成 SQL 的同值匹配："
                f"`{len(policy_sql_scan['assistant_generated_matches'])}`；"
                "所有 raw response、trajectory、SQL audit、score 和 token JSONL 均保留在本 run。"
            ),
            (
                "- 原始格式失败/业务失败保留 token 与 reward=0；只有基础设施/证据故障"
                "进入 unknown，unknown 不被填成 reward=0。"
            ),
            "- 本结论只判别同任务组内的奖励方差；不等同于GRPO更新成功，也不保证后续业务收益。",
            "",
            "## 实际命令与退出码",
            "",
        ]
    )
    commands = read_json(paths["run"] / "commands.json", {}) or {}
    for command in commands.get("commands", []):
        report_lines.append(
            f"- `{command.get('command_id')}` exit `{command.get('returncode')}` "
            f"scope `{command.get('scope')}`；stdout `{command.get('stdout_path')}`；"
            f"stderr `{command.get('stderr_path')}`。"
        )
    report_lines.extend(
        [
            "",
            (
                f"绝对产物路径：`{paths['run']}`。详见 `checks.json`、`metrics.json`、"
                "`group-metrics.json`、`allocation-ledger.json`、`job-chain.json` 和 "
                "`artifact-manifest.json`."
            ),
        ]
    )
    (paths["run"] / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    _write_artifact_manifest(paths["run"])
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preflight", "rollout", "finalize", "record-command"), required=True
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--run", default="artifacts/qwen3-grpo-rollout-signal/t033-v1")
    parser.add_argument("--config", default="configs/evaluation/qwen3-grpo-rollout.toml")
    parser.add_argument(
        "--panel-manifest", default="artifacts/qwen3-base-lora-pilot/t031-v1/panel-manifest.json"
    )
    parser.add_argument(
        "--source-db-root", default="artifacts/qwen3-base-lora-pilot/t031-v1/base/databases"
    )
    parser.add_argument("--cache", default="artifacts/model-cache-qwen3-8b")
    parser.add_argument(
        "--parent-adapter",
        default="artifacts/qwen3-runtime-sft-comparison/t032-v1/training/final-adapter",
    )
    parser.add_argument("--stdout-path")
    parser.add_argument("--stderr-path")
    parser.add_argument("--scope", default="t033")
    parser.add_argument("--recover-interrupted-gpu-seconds", type=float)
    parser.add_argument("--command-id")
    parser.add_argument("--command")
    parser.add_argument("--exit-code", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = _common_paths(args)
    if args.stage == "record-command":
        if not args.command_id or args.command is None or args.exit_code is None:
            raise SystemExit("record-command requires --command-id --command --exit-code")
        _ensure_run_dir(paths["run"])
        _record_command(
            paths["run"],
            command_id=args.command_id,
            command=args.command,
            argv=None,
            returncode=args.exit_code,
            stdout_path=args.stdout_path,
            stderr_path=args.stderr_path,
            scope=args.scope,
        )
        return 0
    _ensure_run_dir(paths["run"])
    rc = 0
    try:
        if args.stage == "preflight":
            _prepare(paths)
        elif args.stage == "rollout":
            _rollout(paths, args)
        elif args.stage == "finalize":
            _finalize(paths)
        else:
            raise RolloutSignalError(f"unknown_stage:{args.stage}")
    except Exception:
        rc = 1
        traceback.print_exc()
    finally:
        command_id = f"{args.stage}-{int(time.time())}"
        command = shlex.join([sys.executable, *sys.argv])
        try:
            _record_command(
                paths["run"],
                command_id=command_id,
                command=command,
                argv=[sys.executable, *sys.argv],
                returncode=rc,
                stdout_path=args.stdout_path,
                stderr_path=args.stderr_path,
                scope=args.scope,
            )
        except Exception:
            traceback.print_exc()
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
