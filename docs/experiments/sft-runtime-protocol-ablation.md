# Runtime 协议消融

历史 train-only 诊断，Job `20391073`，2026-10-17，gpu8Q/gpu803，V100-SXM2 32GB，0:0，31:29。

使用 Qwen2.5-1.5B-Instruct + 早期 E3 Adapter，20 个 train 任务。模型、数据库、工具、评分和预算固定，只改变三项请求协议。

## 分组

| 组 | 初始 observation | 已接受的 Action 历史 | 格式重试 |
| --- | --- | --- | ---: |
| A baseline | 仅首轮 | raw output | 1 |
| B initial-observation | 每轮保留一次 | raw output | 1 |
| C canonical-history | 仅首轮 | canonical JSON | 1 |
| D no-format-retry | 仅首轮 | raw output | 0 |
| E full-alignment | 每轮保留一次 | canonical JSON | 0 |

loader 流式读取 20 条 train 记录后停止，不打开 validation/test 数据库。模型只收到公开任务和实际工具反馈。

## 结果

| 组 | 业务成功 | Parser acceptance | SQL 可执行率 | SQL 重复率 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0/20 | 93.48% | 0% | 79.79% | 9.70 |
| B | 0/20 | 94.27% | 2.06% | 78.35% | 9.85 |
| C | 0/20 | 93.48% | 0% | 79.79% | 9.70 |
| D | 0/20 | 96.15% | 0% | 79.23% | 9.20 |
| E | 0/20 | 93.85% | 2.86% | 70.00% | 6.75 |

保留初始 observation 改变了部分工具选择和 SQL 行为；单独 canonical history 未见独立业务收益。组合方案仍未完成任务。

原报告的 recovery success 使用宽口径；后续严格错误后 SQL 恢复指标仍为零，不能将两者混用。该诊断未训练或更新模型。

原始产物：`artifacts/sft-runtime-ablation/20261017T130311Z-sft-runtime-protocol-ablation/`。后续结果见 [实验汇总](../results.md)。
