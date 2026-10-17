# SFT Runtime Protocol Ablation

状态：`SFT_RUNTIME_ABLATION_COMPLETE`

本实验是 train-only 的闭环诊断，不训练、不运行 GRPO、不重跑正式 validation benchmark，也不访问 test split。固定使用 `train-0`、`train-1` 的 20 条任务和 10 个模板，数据库、Environment、SQLTool、Verifier、Reward、Action budget、base model、E3 adapter、tokenizer、chat template 与 generation 参数保持一致。

## 数据边界

公开任务来自 `artifacts/sft-data-train-teacher-v2/dataset.jsonl`。诊断 loader 不调用 `enterprise_agent_rl.sft_data.load_train_source`，因为该旧 loader 会先对整个 frozen manifest 执行 `json.loads`。新 loader 只流式反序列化 manifest `records` 数组中的 20 条 train 记录，并在每条记录上验证 `split=train` 与 `train-*` ID 后停止；validation/test task payload 不进入对象，也不打开 validation/test 数据库。Reference SQL 仅留在独立 verifier-side `BenchmarkTask` 中，policy 只收到 `PublicTask`。

每次模型请求都会保存完整 messages。诊断 guard 拒绝把 `reference_sql`、`expected_answer` 字段或 verifier reference SQL 注入 system/user 消息；真实 SQL Tool Result 是运行时反馈，不是 gold answer 注入。

## 协议组

| 组 | initial observation | accepted assistant history | format retries |
|---|---|---|---:|
| A `baseline` | 首轮请求保留 | raw output | 1 |
| B `initial-observation` | 每次请求保留一次，位于 task prompt 后、首个 assistant Action 前 | raw output | 1 |
| C `canonical-history` | 首轮请求保留 | 仅规范化已接受 Action，使用 `sft_data.py` serializer | 1 |
| D `no-format-retry` | 首轮请求保留 | raw output | 0 |
| E `full-alignment` | 每次请求保留一次 | canonical JSON | 0 |

诊断 policy wrapper 不改变生产 Runtime 默认行为；三个变量由显式 group config 控制，所有组共用同一闭环代码路径。

## CPU preflight

`preflight.json` 必须记录 train-only loader boundary、有效配置差异、adapter hash、tokenizer/chat-template source hash、协议矩阵、完整 mock episode 检查和 Git 状态。新增测试与 `git diff --check` 必须通过；否则状态为 `SFT_RUNTIME_ABLATION_BLOCKED`，不提交 GPU。

## GPU 作业

GPU 阶段只提交一次 Slurm job，五组顺序执行且一次只加载一个 Qwen engine：

- model `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；
- E3 adapter `artifacts/sft-pilot-20390015/adapter/adapter_model.safetensors`，SHA-256 `a7d252f782a541dfc697055d43c7a25768d99633b6d5c3335131d00a6247bd77`；
- FP16、当前配置的 SDPA math、temperature 0、top-p 1、max tokens 512、seed 17；
- 1 Tesla V100 / CC 7.0、2 CPU、16 GB、60 分钟、`gpuq` QoS；
- 分区按 `gpu2Q → gpu4Q → gpu8Q` 的实测 `squeue`、`sinfo`、账户/QoS `sbatch --test-only` 和近期 `sacct` 证据选择。

产物写入唯一 `artifacts/sft-runtime-ablation/<run-id>/`。每组目录包含 20 条 trajectory/result、完整 request messages、raw generations、SQL audit 与 summary；根目录包含 `preflight.json`、`environment.json`、`comparison.json` 和 Slurm 原始输出。模型、数据库与 checkpoint 不提交 Git。

本次实际运行：run id `20261017T130311Z-sft-runtime-protocol-ablation`，Slurm job `20391073`，`gpu8Q` / `gpu803`，`COMPLETED`、exit code `0:0`、用时 `31:29`。调度选择依据是 `gpu2Q` 的实测预计启动时间明显晚于 `gpu8Q`；`gpu4Q` 无可确认的历史 V100 证据。GPU gate 实际确认 `Tesla V100-SXM2-32GB`、compute capability `7.0`。数据边界记录为 train records deserialized `20`、validation/test task payload `false`、validation/test database opened `false`。

### 结果摘要

| 组 | Success | Parser acceptance | 首轮合法 | 首次 SQL 前 schema/dict | SQL executable | SQL error | Recovery success | SQL repeat | Check | Finish | 平均 tool calls | 平均 latency (s) | 平均 total tokens | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A baseline | 0/20 | 93.48% | 80.00% | 0.00% | 0.00% | 100.00% | 3.72% | 79.79% | 5.00% | 0.00% | 9.70 | 21.31 | 13296.35 | 0 |
| B initial-observation | 0/20 | 94.27% | 80.00% | 10.00% | 2.06% | 97.94% | 9.47% | 78.35% | 55.00% | 5.00% | 9.85 | 15.61 | 16834.35 | 0 |
| C canonical-history | 0/20 | 93.48% | 80.00% | 0.00% | 0.00% | 100.00% | 3.72% | 79.79% | 5.00% | 0.00% | 9.70 | 17.86 | 13291.85 | 0 |
| D no-format-retry | 0/20 | 96.15% | 80.00% | 0.00% | 0.00% | 100.00% | 0.00% | 79.23% | 5.00% | 0.00% | 9.20 | 16.49 | 12026.15 | 0 |
| E full-alignment | 0/20 | 93.85% | 80.00% | 10.00% | 2.86% | 97.14% | 0.00% | 70.00% | 55.00% | 5.00% | 6.75 | 11.83 | 10382.40 | 0 |

所有组的 task success 都为 0，不能仅据此判定协议因素无影响。B 相对 A 在首次 SQL 前 schema/dictionary 覆盖、可执行 SQL、恢复、Check/Finish 和 SQL 重复率上改善，但仍未完成任务；C 的聚合行为与 A 相同，未显示 canonical history 的独立收益；D 没有格式重试且 parser acceptance 上升、token/tool 消耗下降，但恢复成功为 0，体现了重试策略取舍；E 继承 B 的 schema/dictionary 与 Check/Finish 改善，SQL executable 略高且 SQL repeat/tool calls/latency 更低，但恢复仍为 0，因此不支持“组合对齐已足以解决失败”。主要失败信号仍包括错误 SQL 参数/列名、SQL 重复和 action/tool budget exhaustion；训练数据异质性、SQL 参数学习与自主规划能力仍是未消除的解释。

## 解释边界

B/C 相对 A 的提升只支持对应协议因素对当前行为有帮助；D 的 parser failure、重试次数和后续恢复必须一起解释；E 优于单因素组时只说明可能存在互补效果。20-task 结果是诊断性证据；即便 E 仍失败，也不能推出 Runtime 协议没有影响，还需保留训练数据异质性、SQL 参数学习和自主规划能力等解释。
