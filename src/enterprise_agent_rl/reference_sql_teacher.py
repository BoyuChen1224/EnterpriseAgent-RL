"""Train-only, programmatic teacher driven by train reference SQL.

This module is a data-construction policy, not a benchmark policy.  It accepts a
single train ``BenchmarkTask`` and emits actions through the normal episode
runtime.  The reference SQL is used only as SQL supervision; final answers are
assembled by the environment from the actual SQL observation and are later
checked by the independent formal verifier.
"""

# 这是离线 train 数据构建的程序化教师，不是用来参加 Base/SFT/GRPO 评测的策略。
# 它用 train 标签决定 SQL，再通过正常 Runtime 执行；最终答案由真实查询结果组装。
# 不要把“教师能访问 train 标签”误解为业务模型在 validation 推理时也能看到标签。

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from enterprise_agent_rl.agent import Policy, run_episode
from enterprise_agent_rl.common import Action, Decision, Observation, PublicTask, Settings
from enterprise_agent_rl.data import BenchmarkTask
from enterprise_agent_rl.trajectories import StepRecord, Trajectory

TRAJECTORY_SOURCE = "reference_sql_teacher"
POLICY_KIND = "programmatic_reference_sql_teacher"
POLICY_VERSION = "reference-sql-teacher-v1"
LEGACY_ERROR_SQL = "SELECT total_amount FROM orders LIMIT 1"


class ReferenceSQLTeacherError(RuntimeError):
    """Raised when the supervised SQL cannot be completed through the runtime."""


def _decision(kind: str, **arguments: Any) -> Decision:
    return Decision(action=Action(kind=kind, arguments=arguments))


# 只接受 train 任务，按历史观察逐步输出计划、必要工具、SQL、check 和 finish。
# 轨迹会经过正式 Verifier；教师来源必须与真实模型采样来源区分。
class ReferenceSQLTeacherPolicy:
    """A deterministic train-only policy that uses one task's reference SQL."""

    kind = POLICY_KIND
    version = POLICY_VERSION

    def __init__(self, task: BenchmarkTask):
        if task.split != "train":
            raise ValueError("reference_sql_teacher_train_only")
        if not task.reference_sql.strip():
            raise ValueError("reference_sql_teacher_reference_sql_missing")
        self.task = task

    @staticmethod
    def _successful(history: list[StepRecord], kind: str) -> list[StepRecord]:
        return [step for step in history if step.tool_name == kind and step.tool_result.ok]

    @staticmethod
    def _sql_query(step: StepRecord) -> str | None:
        query = step.action.arguments.get("query")
        return query if isinstance(query, str) else None

    def _needs_dictionary(self) -> bool:
        prompt = self.task.public.prompt.lower()
        return "收入" in self.task.public.prompt or "revenue" in prompt

    # 每次只产生下一条 Action，不把参考结果直接写成最终表格。
    # 异常 SQL 或预期步骤失败立即报告，避免把伪造观察当作可监督轨迹。
    def decide(
        self,
        task: PublicTask,
        observation: Observation,
        history: list[StepRecord],
        timeout_s: float,
    ) -> Decision:
        del timeout_s
        if task != self.task.public:
            raise ReferenceSQLTeacherError("teacher_task_mismatch")
        if not history:
            steps = ["读取必要的数据库结构", "执行受监督 SQL", "检查实际查询结果", "提交结果"]
            return _decision("plan", steps=steps)

        last = history[-1]
        if last.tool_name == "schema" and not observation.ok:
            raise ReferenceSQLTeacherError("schema_execution_failed")
        if last.tool_name == "dictionary" and not observation.ok:
            raise ReferenceSQLTeacherError("dictionary_execution_failed")

        if last.tool_name == "sql" and not observation.ok:
            query = self._sql_query(last)
            if self.task.required_sql_error and query == LEGACY_ERROR_SQL:
                return _decision("sql", query=self.task.reference_sql)
            if query == self.task.reference_sql:
                raise ReferenceSQLTeacherError("reference_sql_execution_failed")
            if self.task.required_probe_sql and query == self.task.required_probe_sql:
                raise ReferenceSQLTeacherError("probe_sql_execution_failed")
            raise ReferenceSQLTeacherError("unexpected_sql_execution_failed")

        if last.tool_name == "sql" and observation.ok:
            if observation.result.get("truncated"):
                if self._sql_query(last) == self.task.reference_sql:
                    raise ReferenceSQLTeacherError("reference_sql_result_truncated")
            if self._sql_query(last) == LEGACY_ERROR_SQL and self.task.required_sql_error:
                raise ReferenceSQLTeacherError("required_sql_error_not_observed")

        if last.tool_name == "check":
            if not observation.ok:
                raise ReferenceSQLTeacherError("result_check_failed")
            evidence_step = observation.result.get("evidence_step")
            if type(evidence_step) is not int:
                raise ReferenceSQLTeacherError("result_check_missing_evidence")
            return _decision(
                "finish",
                evidence_step=evidence_step,
                summary="已根据实际 SQL 执行结果完成查询，结果见已检查的证据。",
            )

        if self.task.required_schema and not self._successful(history, "schema"):
            return _decision("schema")

        if self._needs_dictionary() and not self._successful(history, "dictionary"):
            return _decision("dictionary", term="revenue")

        # 受控恢复任务会主动构造一次已知错误，用于提供恢复状态监督。
        # 这不等于自然发生错误后的普遍鲁棒性，评测时应区分两种场景。
        if self.task.required_sql_error:
            legacy_errors = [
                step
                for step in history
                if step.tool_name == "sql"
                and self._sql_query(step) == LEGACY_ERROR_SQL
                and not step.tool_result.ok
            ]
            if not legacy_errors:
                return _decision("sql", query=LEGACY_ERROR_SQL)

        if self.task.required_probe_sql:
            probes = [
                step
                for step in history
                if step.tool_name == "sql"
                and self._sql_query(step) == self.task.required_probe_sql
                and step.tool_result.ok
            ]
            if not probes:
                return _decision("sql", query=self.task.required_probe_sql)

        reference_steps = [
            step
            for step in history
            if step.tool_name == "sql" and self._sql_query(step) == self.task.reference_sql
        ]
        if not reference_steps:
            return _decision("sql", query=self.task.reference_sql)
        if not reference_steps[-1].tool_result.ok:
            raise ReferenceSQLTeacherError("reference_sql_execution_failed")

        final_sql = reference_steps[-1]
        columns = final_sql.tool_result.result.get("columns")
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(column, str) for column in columns)
        ):
            raise ReferenceSQLTeacherError("reference_sql_columns_missing")
        if not any(
            step.tool_name == "check"
            and step.action.arguments.get("evidence_step") == final_sql.step_index
            for step in history
        ):
            return _decision(
                "check",
                evidence_step=final_sql.step_index,
                expected_columns=columns,
            )
        raise ReferenceSQLTeacherError("teacher_state_unreachable")


# 从已保存步骤中确认参考 SQL 是否真实执行，不为统计再调用一次数据库。
def reference_sql_execution_status(trajectory: Trajectory, reference_sql: str) -> dict[str, Any]:
    """Summarize the SQL action's actual runtime observation without re-executing it."""

    matching = [
        step
        for step in trajectory.steps
        if step.tool_name == "sql" and step.action.arguments.get("query") == reference_sql
    ]
    if not matching:
        return {"status": "not_attempted", "ok": False, "truncated": False}
    step = matching[-1]
    if not step.tool_result.ok:
        return {
            "status": "failed",
            "ok": False,
            "truncated": False,
            "error": step.tool_result.error,
            "step_index": step.step_index,
        }
    result = step.tool_result.result
    if result.get("truncated"):
        return {
            "status": "truncated",
            "ok": True,
            "truncated": True,
            "step_index": step.step_index,
        }
    return {
        "status": "success",
        "ok": True,
        "truncated": False,
        "columns": result.get("columns", []),
        "row_count": len(result.get("rows", [])),
        "step_index": step.step_index,
    }


# 给教师轨迹补来源与执行身份；来源元数据不应混入学生可见的公开问题。
def with_reference_sql_provenance(trajectory: Trajectory, task: BenchmarkTask) -> Trajectory:
    """Attach auditable supervision metadata without storing raw reference SQL."""

    return trajectory.model_copy(
        update={
            "policy_metadata": {
                "trajectory_source": TRAJECTORY_SOURCE,
                "policy_kind": POLICY_KIND,
                "source_split": "train",
                "original_supervision_source": "train_task.reference_sql",
                "reference_sql_available": True,
                "reference_sql_sha256": hashlib.sha256(
                    task.reference_sql.encode("utf-8")
                ).hexdigest(),
                "reference_sql_execution": reference_sql_execution_status(
                    trajectory, task.reference_sql
                ),
                "actual_tool_execution": any(step.tool_name for step in trajectory.steps),
                "validation_used_for_training": False,
                "test_used_for_training": False,
            }
        }
    )


# 通过与业务策略相同的 episode 入口运行教师，沿用工具、预算和轨迹记录。
def run_reference_sql_teacher(
    task: BenchmarkTask,
    database: Path,
    settings: Settings,
    audit_path: Path | None = None,
) -> Trajectory:
    """Run one train task through the shared environment and return its raw trace."""

    policy: Policy = ReferenceSQLTeacherPolicy(task)
    trajectory = run_episode(task.public, policy, database, settings, audit_path)
    return with_reference_sql_provenance(trajectory, task)
