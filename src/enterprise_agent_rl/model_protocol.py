"""Bounded syntactic normalization; never invent actions or SQL."""

# 模型输出协议与 chat template 的集中入口，不处理业务执行。
# 先由 parse_action 验证 JSON/参数，再交给环境校验证据 ID、预算等有状态条件。
# 格式容错和业务纠错必须分开：移除完整代码围栏可以，改写模型的 SQL 不可以。

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import Field, ValidationError

from enterprise_agent_rl.common import Action, Contract


# 固定生成和格式重试设置。temperature=0 的冻结评测与随机 rollout 使用不同配置。
class GenerationSettings(Contract):
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    format_retries: int = Field(default=1, ge=0, le=2)
    parse_mode: Literal["strict", "json_fence"] = "json_fence"
    seed: int = Field(default=17, ge=0)
    # None preserves the historical Qwen2.5 template behavior.  Qwen3
    # configs set this explicitly to False so thinking cannot leak into the
    # business Action protocol.
    enable_thinking: bool | None = None


# 统一 SFT 与推理的消息渲染。enable_thinking=None 保留旧 Qwen2.5 行为；
# 显式 False 交给支持该参数的官方模板处理，不用追加字符串冒充硬关闭。
def render_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
):
    """Render one chat through the tokenizer's own template.

    The old Qwen2.5 path deliberately omits ``enable_thinking``.  When a
    newer model opts into a hard switch, a tokenizer that cannot accept the
    switch fails closed instead of silently falling back to a prompt suffix
    such as ``/no_think``.
    """

    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if enable_thinking is not None:
            raise RuntimeError("chat_template_enable_thinking_unsupported") from exc
        raise


# 失败携带已发生的物理尝试，便于保留成本、原始输出及失败类型。
class ModelFailure(RuntimeError):
    def __init__(self, code: str, attempts: list[dict]):
        super().__init__(code)
        self.code, self.attempts = code, attempts


# 返回 (Action, 是否去掉完整 JSON 围栏)。只接受一份完整 JSON，不从杂乱文本中猜答案。
# 未知键、缺参数、空 SQL 和不合法 evidence_step 都在交给工具前拒绝。
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
