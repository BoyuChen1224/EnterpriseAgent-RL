# V100 运行说明

项目在 V100/V100S 32GB、compute capability 7.0 上运行 Qwen2.5-1.5B-Instruct 和 Qwen3-8B。使用 FP16 权重和 SDPA math。

## 固定组合

| 模型 | 环境 | 模型配置 |
| --- | --- | --- |
| Qwen2.5-1.5B-Instruct | Torch 2.2.2+cu121、Transformers 4.46.3 | `configs/model/qwen2.5-1.5b.toml` |
| Qwen3-8B | Torch 2.5.1+cu121、Transformers 4.51.3 | `configs/model/qwen3-8b.toml` |

模型配置显式指定 `float16`，不依赖 `torch_dtype="auto"`。当前路径不使用 BF16 或 FlashAttention-2。

1.5B 早期 FP16 eager attention 出现 QK 溢出，切到 SDPA math 后恢复有限输出，见 [诊断记录](v100-inference-stability.md)。

## 训练

SFT/GRPO 的 SDPA math context 覆盖 forward、backward 和 checkpoint recompute。仅包住 forward，反向重算仍可能选到其他算子。

checkpoint 使用 `use_reentrant=True`，并启用输入梯度。optimizer 只持有 LoRA 参数，Base 冻结；AMP 对非有限梯度执行 skip/backoff，并按同一批次恢复重放。

历史 1.5B GPU 恢复比较中，完整 math scope 得到参数和 optimizer 差异为零。Qwen3 已完成真实更新与独立重载，其 GPU 恢复等价仍未验证。

## Slurm

Qwen3 训练示例申请 1 GPU、4 CPU、64 GB；Base/SFT 并行评测需要两张 GPU、8 CPU、96 GB。1.5B 的资源由各 wrapper 固定。

提交前核对实际 GPU、CPU、内存、时限、账户和 QoS。兼容分区可以切换；替换 pending 作业时先确认旧作业未启动，运行与重试统一计入预算。

`nvcc` 版本、Torch wheel 的 CUDA runtime、节点驱动和 GPU 型号分别记录。排队时间、wrapper wall time、Slurm allocation 也分别统计。

GPU 脚本在计算节点运行，模型从已校验的 cache 离线加载。[Qwen3 入口](qwen3-post-training.md) · [GRPO](grpo-training.md)
