# CPU MVP 对照

2026-10-15，Job `20383968`，freecpuQ / node0857，1 CPU、2 GB，0:0，5 s。两组使用确定性 fixture，没有加载语言模型。

| 指标 | fixture-v1 | fixture-v1-recovery |
| --- | ---: | ---: |
| 任务成功 | 5/5 | 5/5 |
| SQL 可执行 | 6/6 | 6/11 |
| 工具错误 | 0% | 5/26 |
| 平均工具调用 | 4.2 | 5.2 |
| 平均 episode 耗时 | 0.1236 s | 0.1681 s |

recovery 策略在每个任务先故意查询错误字段，再根据反馈修复。结果验证工具和恢复流程，不计入模型能力评测。

本目录的 [正常结果](fixture-report.json)、[恢复结果](recovery-report.json)、[对照](comparison.json) 使用相同 seed、数据库和任务。完整轨迹与日志保存在本地 `artifacts/slurm-evaluation-20383968/`。

当前模型成绩见 [结果](../../docs/results.md)，CPU 测试见 [测试记录](../../docs/validation.md)。
