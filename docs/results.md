# 实验结果

## 总体成绩

业务成功同时检查最终行列、业务步骤和运行预算。所有条目均为 20/20 正式评分，unknown=0。

| 模型 | Base | SFT | SFT + GRPO |
| --- | ---: | ---: | ---: |
| Qwen3-8B | 0/20 | 13/20 | 13/20 |
| Qwen2.5-1.5B-Instruct | 0/20 | 8/20 | 8/20 |

Qwen3 的 Base/SFT 来自 T-032，fresh SFT/GRPO 来自 T-034；1.5B 来自 T-024，并由 T-025 补齐最后一次评分。两套模型的训练配方和实现版本不同，按各自历史实验展示。

## 逐模板对比

每类两个数据库实例，分母均为 2。GRPO 列指 SFT parent 加载其 GRPO delta。

| 任务 | 1.5B Base | 1.5B SFT | 1.5B GRPO | 8B Base | 8B SFT | 8B GRPO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Schema 查询 | 0/2 | 2/2 | 2/2 | 0/2 | 2/2 | 2/2 |
| 区域收入 | 0/2 | 2/2 | 2/2 | 0/2 | 0/2 | 0/2 |
| 客户 Top-3 | 0/2 | 0/2 | 0/2 | 0/2 | 2/2 | 2/2 |
| 指定客户 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 1/2 |
| 指定商品 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| 空结果 | 0/2 | 2/2 | 2/2 | 0/2 | 2/2 | 2/2 |
| 聚合约束 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| 错误恢复 | 0/2 | 0/2 | 0/2 | 0/2 | 2/2 | 2/2 |
| 订单状态 | 0/2 | 2/2 | 2/2 | 0/2 | 2/2 | 2/2 |
| 月度汇总 | 0/2 | 0/2 | 0/2 | 0/2 | 2/2 | 2/2 |

任务使用项目自建的五表 SQLite 数据库，按 train/validation/test 分实例、年份和实体。各划分仍共享 schema 与部分模板，当前结果不代表跨企业数据库泛化。

## Qwen3 SFT 改变了什么

| 指标 | Base | SFT |
| --- | ---: | ---: |
| 业务成功 | 0/20 | 13/20 |
| Plan Action | 149 | 20 |
| 成功 Schema 调用 | 1 | 20 |
| 可执行 SQL / SQL worker | 1/102 | 22/30 |
| 模型请求 | 313 | 112 |
| 平均任务耗时 | 45.24 s | 12.10 s |

54 条 train 决策记录只监督当前 Action。27 次 LoRA 更新后，模型更稳定地读取 Schema、执行 SQL、检查并提交结果。剩余七项失败包括状态值、聚合、字段映射和客户筛选错误。

## GRPO 的结果

Qwen3 采样 4 个任务、每组 4 条 episode；两个组有奖励差异。固定 batch 含 118 条请求，70 条非零优势记录，完成两次更新、140 次 backward。

fresh SFT 与 GRPO 都是 13/20：13 个共同成功，7 个共同失败。111/112 条输出文本相同，20 项 SQL 序列相同。参数更新和重载成功，两次更新没有带来业务增益；尚未测量轨迹概率及合并前后的分布变化。

## 示例怎么读

[README](../README.md) 的 Top-3 和区域收入示例来自 `validation-0-top`、`validation-0-region`。

Top-3 要求先查询订单数。1.5B SFT/GRPO 虽然返回正确排名，却缺少这个步骤，因此 `final_correct=true`、`business_constraints=false`。8B SFT/GRPO 执行了计数和排名两条 SQL，正式评分成功。

区域收入中，8B SFT/GRPO 使用 `status='已支付'`，得到空结果。1.5B SFT/GRPO 使用数据库实际存储的 `'paid'`，返回：

| region | revenue_cents |
| --- | ---: |
| East | 5324500 |
| North | 4549300 |
| South | 3537800 |
| West | 6208800 |

[示例 JSON](examples/model-comparison.json) 保存三个任务、六种配置的 18 条摘要：公开问题、Action 序列、SQL、工具错误、模型最终结果、评分和原轨迹哈希。只摘录已有轨迹，没有重新推理，也没有将私有参考 SQL 当作模型输出。

## 原始结果位置

以下目录位于原实验环境，不随源码分发：

| 实验 | 结果文件 |
| --- | --- |
| 1.5B 三臂结果 | `artifacts/frozen-base-sft-rl-evaluation-closeout/t025-v1/comparison.json` |
| Qwen3 Base/SFT | `artifacts/qwen3-runtime-sft-comparison/t032-v1/comparison.json` |
| Qwen3 SFT/GRPO | `artifacts/qwen3-grpo-update-pilot/t034-v1/metrics/comparison.json` |

每次运行保留配置、任务/数据库指纹、请求/响应、轨迹和评分。早期 SFT 无收益、工程失败和中断记录保存在各自目录，历史结果见 [开发记录](implementation-log.md)。
