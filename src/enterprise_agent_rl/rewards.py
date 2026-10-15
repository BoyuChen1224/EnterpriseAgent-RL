"""Hidden execution verifier. Main reward requires correct evidence and task constraints."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import BenchmarkTask
from enterprise_agent_rl.tools import SQLTool
from enterprise_agent_rl.trajectories import RewardResult, Trajectory, database_fingerprint


def verify(
    task: BenchmarkTask, trajectory: Trajectory, database: Path, settings: Settings | None = None
) -> RewardResult:
    cfg = settings or trajectory.settings
    if task.public != trajectory.task:
        raise ValueError("verifier_task_mismatch")
    if database_fingerprint(database) != trajectory.database_sha256:
        raise ValueError("verifier_database_mismatch")
    expected = SQLTool(database, cfg).execute(task.reference_sql)
    if not expected["ok"] or expected["truncated"]:
        raise RuntimeError("reference_verifier_failed")
    final = trajectory.final_answer
    correct = False
    if final and final.get("columns") == expected["columns"]:
        actual_rows = final.get("rows", [])
        if task.ordered:
            correct = actual_rows == expected["rows"]
        else:

            def canonical(rows):
                return Counter(json.dumps(row, sort_keys=True) for row in rows)

            correct = canonical(actual_rows) == canonical(expected["rows"])
    tools = [s for s in trajectory.steps if s.tool_name]
    sql_steps = [s for s in tools if s.tool_name == "sql"]
    successful_sql = sum(s.tool_result.ok for s in sql_steps)
    errors = sum(not s.tool_result.ok for s in tools)
    fingerprints = [json.dumps(s.action.model_dump(), sort_keys=True) for s in tools]
    repeats = len(fingerprints) - len(set(fingerprints))
    recovered = False
    for i, step in enumerate(sql_steps):
        if not step.tool_result.ok and any(s.tool_result.ok for s in sql_steps[i + 1 :]):
            recovered = True
    within_budget = len(tools) <= cfg.max_tool_calls and len(trajectory.steps) <= cfg.max_actions
    probe_verified = task.required_probe_sql is None
    if task.required_probe_sql is not None:
        probe = SQLTool(database, cfg).execute(task.required_probe_sql)
        if not probe["ok"] or probe["truncated"]:
            raise RuntimeError("reference_probe_failed")
        final_index = final["evidence_step"] if final else -1
        probe_verified = any(
            step.tool_result.ok
            and step.step_index < final_index
            and step.tool_result.result.get("columns") == probe["columns"]
            and step.tool_result.result.get("rows") == probe["rows"]
            for step in sql_steps
        )
    schema_verified = not task.required_schema or any(
        s.tool_name == "schema" and s.tool_result.ok for s in trajectory.steps
    )
    recovery_verified = not task.required_sql_error or recovered
    constraints = (
        within_budget
        and successful_sql >= task.min_sql_calls
        and probe_verified
        and schema_verified
        and recovery_verified
    )
    success = correct and constraints and trajectory.terminal_reason == "completed"
    # Correct tasks stay >=0.9, failed tasks get zero. No incentive for repeated recovery.
    auxiliary = (
        -cfg.step_penalty * len(tools)
        - cfg.repeat_penalty * repeats
        - cfg.error_penalty * errors
        + cfg.recovery_bonus * int(recovered)
    )
    value = min(1.0, max(0.9, 1.0 + max(-0.1, min(0.0, auxiliary)))) if success else 0.0
    reason = "success" if success else "incorrect_result" if not correct else "constraint_violation"
    if trajectory.terminal_reason != "completed":
        reason = trajectory.terminal_reason
    return RewardResult(
        value=value,
        success=success,
        reason=reason,
        components={
            "final_correct": correct,
            "business_constraints": constraints,
            "business_probe_verified": probe_verified,
            "schema_verified": schema_verified,
            "recovery_verified": recovery_verified,
            "within_budget": within_budget,
            "sql_execution_rate": successful_sql / max(1, len(sql_steps)),
            "tool_errors": errors,
            "repeated_calls": repeats,
            "error_recovered": recovered,
            "auxiliary": auxiliary,
        },
    )
