"""One deterministic episode. Receives PublicTask only; never sees a reference answer."""

# 这里是单个任务的状态机：不生成文本，只校验和执行模型提交的 Action。
# 关键状态：history 记录动作/观察，sql_results 按步骤缓存 SQL 结果，checked 保存已检查证据。
# 完成路径是成功 SQL → check(evidence_step) → finish(evidence_step)，最终数据由环境取出。

from __future__ import annotations

import json
import time
from pathlib import Path

from enterprise_agent_rl.common import Action, Observation, PublicTask, Settings
from enterprise_agent_rl.tools import SQLTool


class DataEnvironment:
    def __init__(self, database: Path, settings: Settings, audit_path: Path | None = None):
        self.settings = settings
        self.tool = SQLTool(database, settings, audit_path)
        self.task: PublicTask | None = None

    # 每次新任务重新清空预算和证据，避免上一个任务的 SQL 结果被下一个任务引用。
    def reset(self, task: PublicTask) -> Observation:
        self.task = task
        self.started = time.monotonic()
        self.history: list[tuple[Action, Observation]] = []
        self.tool_calls = 0
        self.terminated = False
        self.final_answer: dict | None = None
        self.sql_results: dict[int, dict] = {}
        self.checked: set[int] = set()
        return Observation(
            step_index=-1,
            ok=True,
            result={
                "task": task.model_dump(),
                "seed": self.settings.seed,
                "max_tools": self.settings.max_tool_calls,
            },
        )

    def _end(self, reason: str, ok: bool = False, result: dict | None = None) -> Observation:
        self.terminated = True
        return Observation(
            step_index=len(self.history),
            ok=ok,
            result=result or {},
            error=None if ok else reason,
            terminated=True,
            reason=reason,
        )

    def stop(self, reason: str) -> Observation:
        return self._end(reason)

    # 先检查 episode/动作预算，再执行动作，最后把剩余预算写入返回观察。
    # 参数错误被转成可见反馈，不直接崩掉整个 Agent；执行后还会再核对截止时间。
    def step(self, action: Action) -> Observation:
        if self.task is None:
            raise RuntimeError("reset_required")
        if self.terminated:
            raise RuntimeError("episode_already_terminated")
        if time.monotonic() - self.started >= self.settings.episode_timeout_s:
            observation = self._end("episode_timeout")
        elif len(self.history) >= self.settings.max_actions:
            observation = self._end("action_budget_exhausted")
        else:
            try:
                observation = self._dispatch(action)
            except (ValueError, TypeError, KeyError) as exc:
                observation = Observation(
                    step_index=len(self.history), ok=False, error=str(exc)[:300]
                )
            if (
                not observation.terminated
                and time.monotonic() - self.started >= self.settings.episode_timeout_s
            ):
                observation = self._end("episode_timeout")
            elif not observation.terminated and len(self.history) + 1 >= self.settings.max_actions:
                observation = self._end("action_budget_exhausted")
        observation = observation.model_copy(
            update={
                "metadata": {
                    "tool_calls_used": self.tool_calls,
                    "remaining_tool_calls": self.settings.max_tool_calls - self.tool_calls,
                    "remaining_actions": self.settings.max_actions - len(self.history) - 1,
                    "seed": self.settings.seed,
                }
            }
        )
        self.history.append((action, observation))
        return observation

    # 按 kind 分派。plan/finish 不增加工具计数，但仍占用一个动作步骤。
    # 其余分支在参数校验前先扣工具次数，错误参数也不能无限免费重试。
    def _dispatch(self, action: Action) -> Observation:
        idx = len(self.history)
        args = action.arguments
        result = {}
        if action.kind == "plan":
            if set(args) != {"steps"} or not isinstance(args["steps"], list) or not args["steps"]:
                raise ValueError("plan_requires_steps")
            result = {"steps": args["steps"]}
        # finish 只能引用通过 check 的 SQL 步骤。summary 是模型文字，rows/columns 来自执行证据。
        elif action.kind == "finish":
            if set(args) != {"evidence_step", "summary"} or not isinstance(args["summary"], str):
                raise ValueError("finish_requires_evidence_step_and_summary")
            evidence = args["evidence_step"]
            if type(evidence) is not int or evidence not in self.checked:
                raise ValueError("finish_requires_checked_sql_evidence")
            result = {
                "summary": args["summary"],
                "evidence_step": evidence,
                "columns": self.sql_results[evidence]["columns"],
                "rows": self.sql_results[evidence]["rows"],
            }
            self.final_answer = result
            return self._end("completed", ok=True, result=result)
        else:
            # 这是调用 SQL worker 之前的预算闸门；被拒绝的 SQL Action 没有实际执行查询。
            if self.tool_calls >= self.settings.max_tool_calls:
                return self._end("tool_budget_exhausted")
            self.tool_calls += 1
            if action.kind == "schema":
                if args:
                    raise ValueError("schema_takes_no_arguments")
                result = self.tool.schema(
                    timeout_s=max(
                        0.001, self.settings.episode_timeout_s - (time.monotonic() - self.started)
                    )
                )
            elif action.kind == "dictionary":
                if set(args) - {"term"}:
                    raise ValueError("invalid_dictionary_arguments")
                result = self.tool.dictionary(args.get("term"))
            elif action.kind == "sql":
                if set(args) != {"query"} or not isinstance(args["query"], str):
                    raise ValueError("sql_requires_query_string")
                result = self.tool.execute(
                    args["query"],
                    context={"task_id": self.task.task_id, "step_index": idx},
                    timeout_s=max(
                        0.001, self.settings.episode_timeout_s - (time.monotonic() - self.started)
                    ),
                )
                if not result["ok"]:
                    return Observation(step_index=idx, ok=False, error=result["error"])
                # 仅成功查询才缓存；查询使用其动作步骤号作为证据 ID，不使用模型自报的结果。
                self.sql_results[idx] = result
            # check 只验证结果完整性和列结构，不提供隐藏答案，也不判断业务语义是否正确。
            elif action.kind == "check":
                if set(args) != {"evidence_step", "expected_columns"}:
                    raise ValueError("check_requires_evidence_and_columns")
                evidence = args["evidence_step"]
                if type(evidence) is not int or evidence not in self.sql_results:
                    raise ValueError("invalid_evidence_step")
                query_result = self.sql_results[evidence]
                if query_result["truncated"]:
                    raise ValueError("cannot_verify_truncated_result")
                if query_result["columns"] != args["expected_columns"]:
                    raise ValueError("unexpected_result_columns")
                # Schema/completeness check only. Hidden correctness is evaluated after termination.
                # 截断结果和列名不匹配的结果不能进入 checked；空结果本身不被一律判错。
                self.checked.add(evidence)
                result = {
                    "evidence_step": evidence,
                    "row_count": len(query_result["rows"]),
                    "complete": True,
                }
        return Observation(step_index=idx, ok=True, result=result)

    def action_fingerprints(self) -> list[str]:
        return [
            json.dumps(a.model_dump(), sort_keys=True)
            for a, _ in self.history
            if a.kind not in ("plan", "finish")
        ]
