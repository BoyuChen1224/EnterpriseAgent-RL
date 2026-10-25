# 项目进展

## 已完成

| 工作 | 结果 |
| --- | --- |
| Runtime、SQL 和轨迹 | 六类 Action、只读查询、错误反馈、预算与回放 |
| 独立评分 | 最终行列、业务约束、必要步骤与 unknown 分开处理 |
| 1.5B SFT / GRPO | 冻结评测 0/20 → 8/20 → 8/20 |
| Qwen3-8B 接入 | V100 FP16 / SDPA math、thinking-off、LoRA 与重载 |
| Qwen3 Runtime 对齐 SFT | 54 条决策、27 次更新；0/20 → 13/20 |
| Qwen3 GRPO pilot | 16 条 rollout、两个非退化组、两次更新；13/20 → 13/20 |
| 本地 API | 1.5B SFT 实际 HTTP/API/SQL/评分 smoke |
| CPU CI | 独立 CPU 依赖、临时数据和模型、格式检查、pytest、smoke |
| Git | 源码统一在 main，阶段提交与 GitHub 同步 |

[结果与示例](results.md) 列出完整对比。历史失败和零收益实验保留。

## 接下来

1. 检查 GRPO 更新前后的轨迹概率、优化方向和 Adapter 合并影响，再决定后续在线训练。
2. 扩充 train 数据库和业务口径，用独立任务检验跨 schema、跨模板泛化。
3. 整理固定训练资产的获取与构建过程，缩小 fresh checkout 到完整复现之间的差距。
4. 增加 Qwen3 Agent API profile，验证持续运行和故障恢复。
5. 评估 BIRD 等外部交互任务的适配，单独管理数据与评分协议。

当前实验使用小型合成库、共享 schema 与部分模板、单 seed。未运行真实模型 test，GRPO 尚无额外业务收益。

## 历史记录

[CPU MVP](validation.md)、[M2A](milestone-2a.md)、[M2B](milestone-2b.md) 和 [早期实验](experiments/base-agent-benchmark.md) 记录当时的状态。日期较早的 Pending 或 0/20 不覆盖后续模型与配置的结果。
