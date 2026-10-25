# 首轮 Base 失败分析

对应 Job `20387111`、commit `857f62b`、20 项 validation。所有任务正式评分，成功 0/20。

## 观察

| 指标 | 数量 |
| --- | ---: |
| SQL / 可执行 SQL | 230 / 16 |
| 工具错误 | 214/234 |
| Schema / Dictionary 调用 | 2 / 0 |
| 无效 Action / 预算耗尽 | 4 / 16 |
| 基础设施错误 | 0 |

SQL 错误中，111 次表不存在、86 次列不存在、1 次语法错误；其余终止信号涉及预算。

## 轨迹中的问题

模型常直接写 SQL，假定存在 `sales`、`payments`、`transactions`、`region`、`order_date` 等表或列。工具返回错误后，模型重复相同假设，没有稳定切换到 Schema 查询。

两条 recovery 轨迹调用了 Schema，但仍未完成正确的查询、Check 和 Finish。可执行 SQL 也没有保证最终业务成功。

本次没有 OOM、非有限 logits、模型服务或 Verifier 故障。高工具错误率主要来自提交的 SQL 不符合当前数据库。

## 后续处理

失败的 validation 轨迹未作为 SFT 监督。训练数据另从 train 生成，并区分确定性教师、reference-SQL 教师和模型轨迹来源，见 [教师覆盖](sft-teacher-coverage-analysis.md)。

原始请求、错误和轨迹保存在 `artifacts/base-agent-benchmark-20387111/`。
