# GRPO 训练

实现位于 `src/enterprise_agent_rl/grpo/`，包括采样、组内优势、response-only loss、LoRA 更新和状态恢复。损失使用本地 `LocalGRPOAdapter`；上游来源及许可证保留在同目录。

## Qwen3 固定批次实验

| 阶段 | 入口 | 配置 |
| --- | --- | --- |
| rollout | `scripts/evaluation/run_qwen3_grpo_rollout.py` | `configs/evaluation/qwen3-grpo-rollout.toml` |
| 更新、重载、对照 | `scripts/training/run_qwen3_grpo_pilot.py` | `configs/training/qwen3-grpo-update-pilot.toml` |

rollout 按 4 个 train task、每组 4 条 episode 采样。16 条均正式评分，两个组有奖励差异。每次请求保存 prompt/completion token IDs、原策略 logprob、采样 logprob、response mask 和 episode 关联。

组内优势为 `(reward - mean) / (sample_std + 1e-4)`。同一 episode 的请求继承该优势，零方差组优势为零。

固定 batch 含 118 条请求：70 条非零优势、48 条零优势。loss 只计算生成 token，归一化使用完整 4745 token。两次更新复用同一 batch 和 old logprobs；实际 140 次 backward、两次成功 optimizer.step。

| 参数 | 值 |
| --- | --- |
| 模型 | Qwen3-8B + 冻结 SFT parent |
| 新 LoRA | rank 8、alpha 16、dropout 0 |
| AdamW learning rate | 2e-6 |
| ratio clipping | epsilon 0.2 |
| KL 系数 | beta 0 |
| AMP | FP16，初始 scale 1024 |

查看入口：

~~~bash
.venv-qwen3/bin/python scripts/training/run_qwen3_grpo_pilot.py --help
~~~

`prepare`/`tiny` 检查批次和 CPU 训练；`train`、`reload`、`compare` 是独立 GPU 阶段；`finalize` 只汇总已有结果。单卡 wrapper 为 `scripts/slurm/qwen3_grpo_update_pilot.sbatch`，用 `T034_STAGE`、`EARL_T034_RUN`、`EARL_T034_CONFIG` 指定阶段、目录和配置。

加载顺序始终是 **fresh Base → SFT parent → GRPO delta**。delta 依赖 parent，不能直接套在原始 Base 上。

fresh SFT/GRPO 均为 13/20，20 项全部已评分，任务成败相同。两次更新与重载已完成，业务成功率仍为 65%。[逐任务结果](results.md)

## 历史 1.5B 在线入口

`scripts/training/run_grpo.py` 支持在线采样和更新，使用独立的 1.5B 配置。不要与 Qwen3 固定批次入口混用。

| 命令 | 用途 |
| --- | --- |
| `validate-config` | 检查 TOML |
| `cpu-demo` | 随机小模型和 fake worker 的 CPU 训练 |
| `train` | 真实在线 rollout、更新及可选评测 |
| `evaluate-frozen` | fresh Base / SFT / GRPO 三臂评测 |
| `cpu-evaluate-frozen` | fake 评测；仍需配置指向的资产 |
| `finalize-frozen`、`finalize` | 离线汇总已有记录 |
| `replay-t021` | 读取指定历史结果 |
| `score-saved` | 固定 T-024/T-025 补评分，不是通用任意任务接口 |

CPU demo，需先安装 [CPU 后训练依赖](reproducibility.md)：

~~~bash
.venv/bin/python scripts/training/run_grpo.py cpu-demo \
  --repo-root "$PWD" \
  --config "$PWD/configs/training/grpo-online.toml" \
  --run "$PWD/artifacts/grpo-cpu-demo"
~~~

`scripts/slurm/grpo_online.sbatch` 默认命令是 `evaluate-frozen`。运行在线训练时，需要显式设置 `EARL_GRPO_COMMAND=train` 和相应配置。

1.5B 已做过连续更新与 GPU checkpoint 恢复等价验证。Qwen3 当前完成更新和独立重载，GPU 恢复等价尚未验证。

## 记录与计量

每个 run 保存 rollout、batch 指纹、训练状态、Adapter 栈和评分。checkpoint 包含 optimizer、scheduler、scaler、RNG 和游标。

业务失败进入成功率分母；缺失响应、评分或关键证据时保留 unknown。SQL Action、SQL worker、Verifier SQL 分别计数。资源以 GPU allocation 累计，排队时间和 wrapper wall time 单列。
