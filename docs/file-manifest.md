# 目录说明

| 目录 | 内容 |
| --- | --- |
| `src/enterprise_agent_rl/` | Runtime、工具、评分、轨迹、模型服务 |
| `src/enterprise_agent_rl/grpo/` | rollout、token、优势、更新、恢复和聚合 |
| `scripts/training/` | SFT、GRPO、LoRA 训练入口 |
| `scripts/evaluation/` | smoke、基准、数据构建和模型对照 |
| `scripts/serving/` | 模型与 Agent API launcher |
| `scripts/diagnostics/` | 历史行为与协议诊断 |
| `scripts/slurm/` | 计算节点作业示例 |
| `configs/` | 模型、Runtime、训练、评测和服务配置 |
| `requirements/` | CPU、CPU 后训练、1.5B GPU、Qwen3 GPU 依赖 |
| `tests/` | 单元测试与 CPU 集成测试 |
| `benchmarks/m2a/` | 合成业务任务的冻结清单 |
| `docs/` | 技术说明、结果和历史报告 |
| `docs/examples/` | 公开的模型输出与结果摘要 |
| `experiments/` | 早期小型汇总记录 |
| `examples/` | API 客户端示例 |

模型、数据库、完整轨迹、checkpoint 和 Slurm 日志保存在本地 `artifacts/`。协作规则和任务提示词也留在本地，不随源码提交。

从 [README](../README.md) 开始；各模块职责见 [架构](architecture.md)，命令和资产见 [复现](reproducibility.md)。
