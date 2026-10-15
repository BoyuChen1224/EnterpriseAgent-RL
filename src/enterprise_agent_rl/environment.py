"""One deterministic episode. Receives PublicTask only; never sees a reference answer."""

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

    def _dispatch(self, action: Action) -> Observation:
        idx = len(self.history)
        args = action.arguments
        result = {}
        if action.kind == "plan":
            if set(args) != {"steps"} or not isinstance(args["steps"], list) or not args["steps"]:
                raise ValueError("plan_requires_steps")
            result = {"steps": args["steps"]}
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
                self.sql_results[idx] = result
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
