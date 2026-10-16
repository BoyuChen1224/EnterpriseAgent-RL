# Milestone 2B preparation evidence

本目录保存不需要收费 GPU 的冻结与审计结果：

- `qwen-cache-verification.json`：固定 Qwen revision 的 snapshot 文件、权重大小和 SHA-256；只在完整权重和无 `.incomplete` blob 时标记 `complete`。
- `experiment-freeze.json`：模型 revision、FP16/eager、生成参数、agent runtime、benchmark manifest 和依赖 lock 的哈希，以及 GPU 授权状态。
- `train-data-audit.json`：20 条 train 任务的 deterministic expert 轨迹覆盖审计。轨迹来源标记为 `deterministic_fixture_expert`，不是 `local_llm` Base 结果。

当前没有 GPU job ID；GPU 验收为 `Pending`。`artifacts/`、模型权重、数据库和运行轨迹仍按仓库规则忽略，不把 CPU fixture、tokenizer 或接口测试替身记为真实模型推理。
