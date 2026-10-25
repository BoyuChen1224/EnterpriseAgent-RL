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
