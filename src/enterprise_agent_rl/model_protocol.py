"""Bounded syntactic normalization; never invent actions or SQL."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import Field, ValidationError

from enterprise_agent_rl.common import Action, Contract


class GenerationSettings(Contract):
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    format_retries: int = Field(default=1, ge=0, le=2)
    parse_mode: Literal["strict", "json_fence"] = "json_fence"
    seed: int = Field(default=17, ge=0)


class ModelFailure(RuntimeError):
    def __init__(self, code: str, attempts: list[dict]):
        super().__init__(code)
        self.code, self.attempts = code, attempts


def parse_action(content: str, mode: str = "json_fence") -> tuple[Action, bool]:
    if not content.strip():
        raise ValueError("empty_output")
    text = content.strip()
    normalized = False
    if mode == "json_fence":
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL)
        if match:
            text, normalized = match[1].strip(), True
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid_json") from exc
    try:
        action = Action.model_validate(value)
    except ValidationError as exc:
        raise ValueError("invalid_action") from exc
    args = action.arguments
    expected = {
        "plan": {"steps"},
        "schema": set(),
        "dictionary": set(args) & {"term"},
        "sql": {"query"},
        "check": {"evidence_step", "expected_columns"},
        "finish": {"evidence_step", "summary"},
    }[action.kind]
    valid = set(args) == expected
    if action.kind == "plan":
        valid &= isinstance(args.get("steps"), list) and bool(args.get("steps"))
        valid &= isinstance(args.get("steps"), list) and all(
            isinstance(s, str) and bool(s.strip()) for s in args["steps"]
        )
    if action.kind == "dictionary" and "term" in args:
        valid &= isinstance(args["term"], str)
    if action.kind == "sql":
        valid &= isinstance(args.get("query"), str) and bool(str(args.get("query", "")).strip())
    if action.kind in ("check", "finish"):
        valid &= type(args.get("evidence_step")) is int and args["evidence_step"] >= 0
    if action.kind == "check":
        cols = args.get("expected_columns")
        valid &= isinstance(cols, list) and bool(cols)
        valid &= isinstance(cols, list) and all(isinstance(c, str) for c in cols)
    if action.kind == "finish":
        valid &= isinstance(args.get("summary"), str)
    if not valid:
        raise ValueError("invalid_action_arguments")
    return action, normalized
