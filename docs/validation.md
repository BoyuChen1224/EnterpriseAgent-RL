# 测试记录

## 当前 CPU CI

2026-10-25，commit `e2b8100` 的 [CPU checks](https://github.com/BoyuChen1224/EnterpriseAgent-RL/actions/runs/40168667152) 全部通过：

- Ruff check、format check、CPU smoke 通过。
- pytest：**197 passed、2 skipped、5 warnings，70.17 s**。
- 两个 skip 对应 Git-ignored 历史实验包；单元测试使用临时数据和随机小模型。

依赖与复现命令见 [运行说明](reproducibility.md)。这组测试不加载真实 Qwen 权重，也不替代模型能力评测。

覆盖工具只读限制、预算与超时、Action 解析、轨迹回放、数据 mask、LoRA 梯度、checkpoint 恢复、HTTP/进程生命周期和失败结果保存。

## CPU MVP 历史记录

2026-10-15，在 freecpuQ 计算节点完成：

| Job | 节点 | 结果 |
| --- | --- | --- |
| 20383967 | node0857 | 48 passed，CPU smoke 通过；0:0，41 s |
| 20383968 | node0857 | 正常/恢复 fixture 均 5/5；0:0，5 s |
| 20384068 | node0928 | 69 passed；0:0，41 s |
| 20384095 | node0858 | 69 passed，CPU smoke 通过；0:0，37 s |

这些任务使用确定性策略。原始数据见 [CPU MVP](../experiments/cpu-mvp/README.md)。

开发时曾出现 45 passed、2 failed，以及恢复对照 4/5：同步 Schema 读取耗时 46.6 s，触发 episode timeout。将 Schema 查询移入有硬超时的子进程后，重新在计算节点通过测试。失败记录保留在本地 `artifacts/release/`。

## 模型实验

模型的成功率单独记录在 [结果](results.md)。CPU smoke 中的 fixture 成功、随机小模型 backward 和 HTTP mock 均按测试结果记录，不计入 Base/SFT/GRPO 成绩。
