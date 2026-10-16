# Base Agent validation failure analysis

本记录对应 Base Agent Benchmark job `20387111`，其 benchmark artifact 记录的
commit 为 `857f62b`，split 为 `validation`，任务数为 20。本文只分析已有
artifact，不修改 Base benchmark 的 prompt、tool schema、reward、generation
configuration 或评测配置。

## 结论摘要

Base Agent 的主要失败不是基础设施故障，而是模型在真实工具反馈下没有建立
schema-first 的查询策略，并在 SQL 失败后反复尝试相同的错误假设。20 条任务全部
被 verifier 评分，但成功率为 `0/20`。工具执行本身提供了明确的 SQLite 错误反馈，
模型通常没有据此切换到 schema discovery 或修正列/表名。

| 指标 | 观察值 |
| --- | ---: |
| Task Success Rate | `0/20` |
| SQL calls | `230` |
| Executable SQL calls | `16/230` (`6.9565%`) |
| Tool errors | `214/234` (`91.4529%`) |
| `schema` calls | `2` |
| `dictionary` calls | `0` |
| Verifier-scored episodes | `20/20` |
| Infrastructure failures | `0` |
| Action Budget Exhausted | `16` |
| Invalid Action | `4` |

SQL 失败的业务错误分布为 `no_such_table=111`、`no_such_column=86`、
`syntax_error=1`。剩余的 SQL/tool 终止信号来自工具预算或 action 预算耗尽。

## 逐条轨迹证据

`benchmark-report.json` 与 20 个 `task-*-trajectory.json` 显示：

- 除两条 recovery 轨迹外，任务在第一次决策就直接进入 SQL；普通任务没有先调用
  `schema`，全部 20 条轨迹没有 `dictionary` 调用。
- `validation-0/1-region` 反复查询不存在的 `region` 列；
  `customer` 轨迹反复尝试不存在的 `transactions`、`customer_data` 等表；
  `monthly` 轨迹反复尝试不存在的 `payments` 表；`product` 轨迹反复尝试
  `sales` 表和一个语法错误查询；`status` 轨迹反复尝试不存在的 `order_date`。
- 两条 recovery 轨迹至少执行了 schema，并收到了 `no such column: payment_date`
  的真实反馈，但随后反复产生 SQL/finish 组合，直到 action budget 耗尽，没有形成
  成功的最终证据。
- 4 条 `Invalid Action` 轨迹没有进入有效工具执行，原始生成历史还包含
  `invalid_action`、`invalid_json` 和 `invalid_action_arguments` 格式问题。
- 16 条 `tool_budget_exhausted` 轨迹在错误反馈后继续重复或轻微改写错误 SQL，
  说明 recovery loop 是模型行为的后果，而不是工具执行器自行重试。

## 失败归因

### Model behavior failure

1. **Schema discovery 被跳过。** `schema:2`、`dictionary:0`，而 SQL:230；除
   recovery 任务以外的任务几乎都没有读取数据库结构或业务口径。
2. **表名和列名幻觉。** 错误集中在不存在的业务表（如 `sales`、`payments`、
   `transactions`）和不存在的列（如 `region`、`order_date`、`c.customer_name`）。
3. **反馈利用不足。** 得到 `no_such_table`/`no_such_column` 后没有基于 schema 或
   错误信息收敛到当前五表数据库，而是保持同一假设并消耗预算。
4. **终止与证据策略失败。** 模型在可执行 SQL 后没有稳定完成 `check` → `finish`，
   或在错误状态下继续生成 finish/SQL，导致 action budget 或 tool budget 终止。
5. **协议鲁棒性不足。** 4 条轨迹以 Invalid Action 结束，生成尝试还出现无效 JSON
   或不符合 Action 参数约束的输出。

这些结论是对实际 action、observation 和 model-attempt 记录的行为归纳，不把
reference SQL 或 oracle 输出输入到模型上下文，也不把失败归因扩大为模型之外的
因素。

### Environment/tool failure

SQLTool 返回的错误与 SQLite 数据库实际 schema 一致，错误反馈是可解释的业务层
信号；工具预算和 action 预算也按既定 Settings 执行。`schema`、`check` 和 verifier
在实际被调用的轨迹中能够返回结果。因而本次没有证据表明 SQLTool、环境状态或
verifier 产生了基础设施性质的错误。工具错误率很高，但这里的“tool error”主要是
模型提交了不可执行 SQL，不等同于工具实现故障。

### Infrastructure failure

本次 Base artifact 明确记录：

- verifier errors: `0`
- runtime errors: `0`
- non-finite logits: `false`
- OOM/CUDA/tokenizer/inference infrastructure errors: `0`
- 20/20 episode 均完成独立 verifier scoring

因此 `infrastructure_failure=0`。GPU/model load、依赖、数据库和 verifier 的
可用性不是本轮 0/20 成功率的主要解释。

## 边界与后续数据决策

当前 Base validation 轨迹全部属于 `local_llm`，且成功数为零，不能直接转换成
SFT supervision。现有 train-only deterministic audit 另行标记为
`deterministic_fixture_expert`；后续管线必须继续保持两种 provenance 分离，只接受
真实经过 Tool execution、现行 verifier 和正式 reward 的 train 轨迹。

本分析不启动 SFT/GRPO，不申请 GPU，也不使用 validation/test 轨迹构造训练数据。
