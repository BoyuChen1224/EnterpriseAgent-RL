# FP16 attention 溢出诊断

历史诊断使用 Qwen2.5-1.5B-Instruct、Torch 2.2.2+cu121、Transformers 4.46.3。Job `20385898` 在 V100S 32GB 上完成。

## 现象与原因

FP32 eager 输出有限；FP16 eager 在第一层 `attention_scores_qk` 出现非有限值。Q、K 本身有限，FP16 QKᵀ 溢出产生 858 个 +Inf、195 个 -Inf，随后 softmax 和最终 logits 出现 NaN。

## 对照

| 路径 | 结果 |
| --- | --- |
| FP32 eager | 有限，36.88 token/s |
| FP16 eager | QK scores 和 logits 非有限 |
| FP16 + SDPA math | finite ratio 1.0，44.40 token/s |

该次记录的峰值 reserved VRAM 约 13.15 GB。速度和显存仅对应此诊断配置，不作为 Qwen3 或其他 batch 的性能结论。

## 修复

模型配置使用 `attention_backend="sdpa_math"`，映射到 Transformers 的 `attn_implementation="sdpa"`。运行时局部启用 math，关闭 flash 和 mem-efficient。

加载和生成均设置 `output_attentions=False`，避免 Qwen2 为返回 attention weights 而退回 eager。FP32 attention-score accumulation 保留为诊断参考。

## 故障记录

每个生成步骤在选 token 前检查 logits。发现 NaN/Inf 时抛出 `non_finite_logits`，记录 step、shape、dtype、device、finite ratio、NaN/Inf 数量和模型身份。

正常事件记录 token、耗时、吞吐、allocator 峰值和实际 backend。非有限值不替换为零，也不通过改采样参数掩盖。

训练时还需要把 math context 覆盖到 backward/recompute，见 [V100 运行说明](v100-compatibility.md)。
