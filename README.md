# EnterpriseAgent-RL

[![CPU checks](https://github.com/BoyuChen1224/EnterpriseAgent-RL/actions/workflows/cpu.yml/badge.svg?branch=main)](https://github.com/BoyuChen1224/EnterpriseAgent-RL/actions/workflows/cpu.yml)

一个面向企业数据分析的 Agent 后训练项目。模型通过工具查询 SQLite，保留每轮交互，再用这些轨迹做 LoRA SFT 和 GRPO。

实现包括多轮 Runtime、只读 SQL、独立业务评分、轨迹回放、训练和本地 API。Qwen3-8B 已在 V100 32GB 上完成 SFT、GRPO 更新和重载，使用 FP16、SDPA math，关闭 thinking。

## Base / SFT / GRPO 对比

每项实验评测 20 个 validation 任务。成功要求答案正确、满足业务约束，并在预算内完成。

| 模型 | Base | SFT | SFT + GRPO | GRPO 相对 SFT |
| --- | ---: | ---: | ---: | ---: |
| **Qwen3-8B** | 0/20 · 0% | **13/20 · 65%** | **13/20 · 65%** | 0 |
| **Qwen2.5-1.5B-Instruct** | 0/20 · 0% | **8/20 · 40%** | **8/20 · 40%** | 0 |

Base 指未加载本项目 Adapter 的模型。两套实验使用相同的 20 个任务，但训练配方、依赖和评测实现版本不同；这张表展示已完成结果，不把差值单独归因于模型大小。Qwen3 的 Base/SFT 和 SFT/GRPO 分别做了配对评测，所有任务均已评分。

Qwen3 SFT 使用 54 条逐决策样本、27 次更新。GRPO 从 16 条 episode 中采到两个有奖励差异的任务组，复用固定批次做了两次更新，业务成功率没有继续提高。数据是小规模合成库，共享 schema 和部分模板；尚未运行真实模型 test。

## 真实任务示例

以下摘自已保存的模型轨迹，没有重新生成回答。完整逐模板结果和 SQL 见 [结果说明](docs/results.md)，18 条示例摘要见 [JSON](docs/examples/model-comparison.json)。

### 客户消费 Top-3

> 先查询 2023 年已支付订单数量（列 paid_orders），再列消费最高的前 3 名客户，返回 customer, revenue_cents，收入降序、客户名升序。

| 模型 | Base | SFT | SFT + GRPO |
| --- | --- | --- | --- |
| Qwen3-8B | 使用不存在的字段，耗尽工具预算 | **成功：先计数，再查询 Top-3** | **成功：与 SFT 相同** |
| Qwen2.5-1.5B-Instruct | 重复查询不存在的 `customer` 列 | 排名正确，但漏掉订单数查询，业务失败 | 同 SFT，仍未满足业务要求 |

Qwen3 SFT 的两次 SQL 查询如下：

~~~sql
SELECT COUNT(*) AS paid_orders
FROM orders
WHERE status='paid'
  AND ordered_at>='2023-01-01' AND ordered_at<'2024-01-01';

SELECT c.name AS customer,
       SUM(i.unit_price_cents*i.quantity) AS revenue_cents
FROM orders o
JOIN order_items i ON o.id=i.order_id
JOIN customers c ON o.customer_id=c.id
WHERE o.status='paid'
  AND o.ordered_at>='2023-01-01' AND o.ordered_at<'2024-01-01'
GROUP BY c.id
ORDER BY revenue_cents DESC,c.name ASC
LIMIT 3;
~~~

返回结果，金额单位为分：

| customer | revenue_cents |
| --- | ---: |
| validation-0-Customer-04 | 2530700 |
| validation-0-Customer-08 | 2390100 |
| validation-0-Customer-07 | 2201600 |

### 区域收入

> 统计 2023 年各区域已支付订单收入（分），返回 region, revenue_cents，按 region 排序。

| 模型 | Base | SFT | SFT + GRPO |
| --- | --- | --- | --- |
| Qwen3-8B | 错误字段，耗尽动作预算 | `status='已支付'`，查询为空，业务失败 | 同 SFT，仍失败 |
| Qwen2.5-1.5B-Instruct | 错误字段，耗尽工具预算 | **成功：使用 `status='paid'`，返回四个区域** | **成功：与 SFT 相同** |

8B 的整体成绩更高，但这个任务仍输给 1.5B。两套模型的 GRPO 都没有修正各自 SFT 的这些失败。

## 核心架构与代码导读

模型决定下一步，Runtime 执行动作，Verifier 在任务结束后独立评分。训练只使用 train 轨迹，评测参考 SQL 和答案不进入模型输入。

~~~mermaid
flowchart LR
  U[公开任务 / 任务 API] --> R[Agent 决策循环]
  R --> P[模型 Policy]
  P --> R
  R --> E[Environment 状态机]
  E --> T[Schema / Dictionary / SQL]
  T --> D[(只读 SQLite)]
  E --> H[轨迹与请求记录]
  H --> V[独立 Verifier / 评测]
  H -->|train| S[SFT 数据与训练]
  H -->|train| G[GRPO 采样与更新]
  S --> P
  G --> P
~~~

### 核心文件

以下文件已补充中文注释。建议先顺着一次任务读，再进入训练代码。

| 模块 | 文件 | 主要看什么 |
| --- | --- | --- |
| 数据契约 | [common.py](src/enterprise_agent_rl/common.py)、[data.py](src/enterprise_agent_rl/data.py) | `PublicTask`、`Action`、`Observation`；公开问题与私有评分信息的边界 |
| 决策循环 | [agent.py](src/enterprise_agent_rl/agent.py) | `run_episode` 主循环、`LocalModelPolicy.decide` 消息构建与格式重试 |
| 环境状态机 | [environment.py](src/enterprise_agent_rl/environment.py) | 动作/工具预算、SQL 证据、`check → finish`；最终数据来自实际查询 |
| 模型协议 | [model_protocol.py](src/enterprise_agent_rl/model_protocol.py) | chat template、thinking 开关、JSON/参数校验；格式容错不改写业务 SQL |
| 工具执行 | [tools.py](src/enterprise_agent_rl/tools.py)、[sql_worker.py](src/enterprise_agent_rl/sql_worker.py) | 父进程超时、只读连接、操作白名单、查询资源限制 |
| 轨迹与回放 | [trajectories.py](src/enterprise_agent_rl/trajectories.py) | `StepRecord` 前后观察、真实模型输出、原子保存、动作回放 |
| 评分与指标 | [rewards.py](src/enterprise_agent_rl/rewards.py)、[evaluation.py](src/enterprise_agent_rl/evaluation.py)、[frozen_evaluation.py](src/enterprise_agent_rl/grpo/frozen_evaluation.py) | 结果/业务步骤/预算检查；业务失败与技术 unknown 分开 |
| 基准与教师 | [benchmark.py](src/enterprise_agent_rl/benchmark.py)、[reference_sql_teacher.py](src/enterprise_agent_rl/reference_sql_teacher.py) | 合成实例、Python oracle、train 教师的实际工具执行 |
| SFT 数据 | [sft_data.py](src/enterprise_agent_rl/sft_data.py)、[sft_training.py](src/enterprise_agent_rl/sft_training.py) | 来源筛选、逐决策上下文、重新分词、assistant-only labels |
| SFT 更新 | [qwen3_sft.py](src/enterprise_agent_rl/qwen3_sft.py)、[lora.py](src/enterprise_agent_rl/lora.py) | 54 条样本、梯度累积、AMP 重放；LoRA 注入/恢复/合并的区别 |
| 数值稳定性 | [inference_stability.py](src/enterprise_agent_rl/inference_stability.py) | SDPA math 作用域、非有限 logits；训练须覆盖 backward 重算 |
| GRPO 数据 | [contracts.py](src/enterprise_agent_rl/grpo/contracts.py)、[rollout.py](src/enterprise_agent_rl/grpo/rollout.py)、[qwen3_rollout.py](src/enterprise_agent_rl/grpo/qwen3_rollout.py) | episode 与物理请求、原生 token/logprob、同任务分组奖励 |
| GRPO 损失与批次 | [upstream.py](src/enterprise_agent_rl/grpo/upstream.py)、[qwen3_update.py](src/enterprise_agent_rl/grpo/qwen3_update.py) | 组内优势、概率比 clipping、response mask、固定完整 token 分母 |
| GRPO 更新与恢复 | [torch_backend.py](src/enterprise_agent_rl/grpo/torch_backend.py)、[trainer.py](src/enterprise_agent_rl/grpo/trainer.py)、[state.py](src/enterprise_agent_rl/grpo/state.py) | Base→SFT parent→delta；optimizer/scaler/RNG/游标与连续更新 |
| 运行记录与预算 | [evidence.py](src/enterprise_agent_rl/grpo/evidence.py)、[budget.py](src/enterprise_agent_rl/grpo/budget.py) | 请求/响应/事件绑定、评分资格、累计 GPU 与调用上限 |
| 模型 HTTP | [model_server.py](src/enterprise_agent_rl/model_server.py) | 权重加载、模型身份、`/v1/chat/completions`、生成日志 |
| 任务 API 与生命周期 | [serving.py](src/enterprise_agent_rl/serving.py)、[service_config.py](src/enterprise_agent_rl/service_config.py)、[service_lifecycle.py](src/enterprise_agent_rl/service_lifecycle.py) | 任务提交/状态/轨迹、profile 核验、全局 deadline 和进程回收 |

### 阅读顺序与运行入口

1. 从 `common.py` 和 `agent.py::run_episode` 理解一次决策，再跟到 `environment.py::_dispatch`。
2. 读 `tools.py → sql_worker.py`，看动作如何变成受限查询；回到 `trajectories.py → rewards.py` 看结果如何保存和判分。
3. 读 `sft_training.py::format_sft_example → qwen3_sft.py::train_lora_schedule`，看哪些 token 参与 loss、27 次更新怎样计算。
4. 读 `grpo/contracts.py → rollout.py → upstream.py → torch_backend.py::QwenPolicy.update`，理解奖励怎样变成梯度。
5. 对照 `qwen3_rollout.py` 和 `qwen3_update.py`：当前 Qwen3 先采样，再复用固定批次两次更新；历史 1.5B 在线入口另行保留。
6. 最后读 `model_server.py → serving.py`，看模型推理 HTTP 与任务 API 怎样连接，避免把两层服务混为一谈。

| 运行阶段 | 实际入口 | 说明 |
| --- | --- | --- |
| Qwen3 SFT | [run_qwen3_sft.py](scripts/training/run_qwen3_sft.py) | `prepare / tiny / train / reload`；CPU 小模型与真实训练复用核心循环 |
| Qwen3 分组采样 | [run_qwen3_grpo_rollout.py](scripts/evaluation/run_qwen3_grpo_rollout.py) | train-only rollout 与评分，不进行参数更新 |
| Qwen3 GRPO | [run_qwen3_grpo_pilot.py](scripts/training/run_qwen3_grpo_pilot.py) | `prepare / tiny / train / reload / compare / finalize`，保留 SFT parent 依赖 |
| 本地任务服务 | [run_sft_n_api.py](scripts/serving/run_sft_n_api.py) | `smoke / serve`；现成 profile 绑定 1.5B SFT N，Qwen3 已验证 HTTP Runtime，任务服务 profile 尚待接入 |

具体资产、命令和资源要求见下方文档。阅读源码或运行 `--help` 不需要启动 GPU 实验。

## 快速开始

CPU smoke 使用临时数据库和确定性策略，检查工具、评分、轨迹回放及 API：

~~~bash
git clone --branch main https://github.com/BoyuChen1224/EnterpriseAgent-RL.git
cd EnterpriseAgent-RL
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/cpu-py312-linux.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python scripts/evaluation/cpu_smoke.py
~~~

完整 CPU 测试另装 `requirements/cpu-training-py312-linux.lock`。GPU 运行需要模型、Adapter 和数据，步骤见 [Qwen3 训练指南](docs/qwen3-post-training.md)。

## 文档与入口

| 内容 | 文档 | 主要入口 |
| --- | --- | --- |
| 模块与数据流 | [架构](docs/architecture.md) | `src/enterprise_agent_rl/` |
| 两套模型的实验结果 | [结果](docs/results.md) | `docs/examples/model-comparison.json` |
| SFT、重载、冻结评测 | [Qwen3 训练](docs/qwen3-post-training.md) | `scripts/training/run_qwen3_sft.py` |
| rollout 与策略更新 | [GRPO](docs/grpo-training.md) | `scripts/training/run_qwen3_grpo_pilot.py` |
| 本地模型与任务 API | [服务](docs/model-serving.md) | `scripts/serving/run_sft_n_api.py` |
| 环境、测试和资产 | [复现](docs/reproducibility.md) | `requirements/` |
| 后续工作 | [路线图](docs/roadmap.md) | — |

源码统一维护在 `main`。模型、数据库、checkpoint 和原始运行日志保存在本地，不随仓库提交。
