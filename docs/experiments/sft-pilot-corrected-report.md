# 首轮 SFT 修正运行

历史 1.5B E3 实验：训练完成，但业务成功率没有改善。该 Adapter 与后续 54 条逐决策训练得到的 N 不同。

## 输入与训练

| 项目 | 记录 |
| --- | --- |
| 模型 | Qwen2.5-1.5B-Instruct，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| 训练代码 | `408590d64a3f1d8d53c46dbe54c267c0636543d4` |
| 数据 | 20 条 train 轨迹，SHA-256 `e3a583be6fd7b399e9088f098f317d7589bf564dcce506b0d8a5de825ea40e3c` |
| Job | `20390015`，gpu2Q/gpu207，V100S 32GB |
| 参数 | 3 epochs，batch 1，累积 4，learning rate 1e-4，seed 42 |
| LoRA | rank 8、alpha 16、dropout .05，196 modules，392 tensors |
| 训练参数量 | 9,232,384；Base 参数冻结 |
| 更新 | 15/15 |
| epoch mean loss | 1.064962 → 0.688628 → 0.550508 |

序列 771–1732 token，无截断。训练 logits、loss 和梯度有限，无 OOM/CUDA 错误；保存了 Adapter、optimizer、scheduler、scaler 和 RNG。

## Validation

Job `20390127` 在 gpu8Q/gpu801 完成加载、196 模块 merge 和 20 项评分，exit 0。

| 指标 | Base | E3 SFT |
| --- | ---: | ---: |
| 业务成功 | 0/20 | 0/20 |
| 可执行 SQL | 16/230 | 0/75 |
| 平均工具调用 | 11.70 | 3.75 |
| 平均任务耗时 | 43.2186 s | 19.3827 s |
| 平均输入 / 输出 token | 17264.75 / 1061.4 | 5708.10 / 625.70 |
| 无效 Action / 预算耗尽 | 4 / 16 | 15 / 5 |

两臂均无基础设施错误。loss 下降与交互减少没有转化为业务成功。

## 产物

- 训练：`artifacts/sft-pilot-20390015/`。
- 验证：`artifacts/sft-agent-benchmark-20390127/`。
- Adapter SHA-256：`a7d252f782a541dfc697055d43c7a25768d99633b6d5c3335131d00a6247bd77`。
- 旧失败 Job `20389950` 的目录保留。

此后开展了 [Runtime 协议诊断](sft-runtime-protocol-ablation.md) 和逐决策 SFT。当前两套模型的结果见 [汇总](../results.md)。
