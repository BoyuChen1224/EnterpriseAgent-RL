# M2B 准备记录

2026-10-16，保存 1.5B 模型缓存和 train 来源检查。当时未提交 GPU 作业。

| 文件 | 内容 |
| --- | --- |
| [qwen-cache-verification.json](qwen-cache-verification.json) | revision、8 个 snapshot 文件、权重大小与 SHA-256，缓存完整 |
| [experiment-freeze.json](experiment-freeze.json) | 当时的模型、生成、Runtime 和基准配置；包含历史 eager 预案 |
| [train-data-audit.json](train-data-audit.json) | 确定性教师 20 项 train 审计，6 成功、14 失败 |

这些 JSON 和原始调度检查输出保持原样。当前使用 FP16 + SDPA math；后续运行见 [V100](../../docs/v100-compatibility.md)，模型结果见 [汇总](../../docs/results.md)。
