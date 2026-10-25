# 首轮 SFT 训练失败

历史 Job `20389950`，状态 `SFT_TRAINING_FAILED`。模型加载后、第一次 optimizer 更新前退出。

| 项目 | 记录 |
| --- | --- |
| 模型 | Qwen2.5-1.5B-Instruct，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| 代码 | `fb3dc863b2d0c0508154e8ee7a75c5849d56e2b6` |
| 数据 | 20 条 train 轨迹，6 条确定性教师、14 条 reference-SQL 教师 |
| 数据 SHA-256 | `e3a583be6fd7b399e9088f098f317d7589bf564dcce506b0d8a5de825ea40e3c` |
| 环境 | Torch 2.2.2+cu121、Transformers 4.46.3、FP16、SDPA math |
| 资源 | gpu2Q/gpuq，gpu207，1 V100S 32GB、2 CPU、16 GB |
| 运行 | 2m38s，exit 1:0 |

错误为 `no_trainable_lora_parameters`：LoRA 参数实际命名为 `lora_A` / `lora_B`，选择器却查找 `lora_A.weight` / `lora_B.weight`。

修复 commit `d2d2fcb` 后，CPU 回归 89 passed、3 skipped，静态检查通过。该 Job 没有产生更新、Adapter 或验证集结果。

原包为 `artifacts/sft-pilot-20389950/`，后续新运行单列在 [修正训练报告](sft-pilot-corrected-report.md)。
