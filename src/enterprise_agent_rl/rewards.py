"""Hidden execution verifier. Main reward requires correct evidence and task constraints."""

# 独立业务评分入口：只有这里接收含参考 SQL 的 BenchmarkTask。
# 环境中的 check 仅验形状；这里同时验证最终数据、业务必要步骤和正常终止。
# SQL 可执行、JSON 合法、loss 下降，都不能直接替代本模块的任务成功条件。

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import BenchmarkTask
from enterprise_agent_rl.tools import SQLTool
from enterprise_agent_rl.trajectories import RewardResult, Trajectory, database_fingerprint


# 先核对任务与数据库身份，再在评分侧执行参考 SQL，并与轨迹中的真实最终结果比较。
# 参考查询失败属于评分基础设施/数据问题，应抛错，不当作模型业务失败。
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
        # 有序任务逐行比较；无序任务用 Counter 比较多重集合，避免 set 丢掉重复行。
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
    # 此 recovered 是“错误后某次 SQL 成功”的宽口径，不是“下一次 SQL 成功”的严格恢复指标。
    # 也不代表修复后的查询满足业务需求；诊断报告应另外明确分母和恢复定义。
    recovered = False
    for i, step in enumerate(sql_steps):
        if not step.tool_result.ok and any(s.tool_result.ok for s in sql_steps[i + 1 :]):
            recovered = True
    within_budget = len(tools) <= cfg.max_tool_calls and len(trajectory.steps) <= cfg.max_actions
    probe_verified = task.required_probe_sql is None
    # 必要前置查询按执行结果验证，并要求发生在最终证据步骤之前；不是比 SQL 字符串。
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
    # 任务成功要求：最终数据正确、必要步骤/预算合格、终止原因为 completed，三者同时成立。
    success = correct and constraints and trajectory.terminal_reason == "completed"
    # Correct tasks stay >=0.9, failed tasks get zero. No incentive for repeated recovery.
    # 辅助项只在主任务成功时影响分值；失败仍为零，不允许靠反复造错恢复取得成功奖励。
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
