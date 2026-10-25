# 首轮 1.5B Base 评测

历史实验：Qwen2.5-1.5B-Instruct，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，FP16 + SDPA math。

Job `20387111` 使用 gpu2Q/gpuq、1 V100、2 CPU、16 GB，完成 20 项 validation。所有任务均被独立 Verifier 评分，业务成功 0/20，基础设施错误为零。

| 指标 | 结果 |
| --- | ---: |
| SQL 可执行 | 16/230，6.9565% |
| 工具错误 | 214/234，91.4529% |
| 平均工具调用 | 11.7 |
| 平均任务耗时 | 43.2186 s |
| 平均输入 / 输出 token | 17264.75 / 1061.4 |
| 终止原因 | 4 项无效 Action，16 项预算耗尽 |

未访问真实模型 test。完整产物位于 `artifacts/base-agent-benchmark-20387111/`。

[失败分析](base-agent-failure-analysis.md) 解释 Schema 缺失和错误循环。后续 1.5B SFT 与当前 Qwen3 的成绩见 [结果](../results.md)，它们使用不同阶段的模型与配置。
