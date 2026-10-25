# 架构

Runtime 负责执行，Policy 负责决策。训练程序消费轨迹，不接管工具或评分逻辑。

~~~mermaid
flowchart LR
  T[公开任务] --> R[Runtime]
  R --> P[模型 Policy]
  P --> R
  R --> E[Environment]
  E --> S[Schema / Dictionary / SQL]
  S --> D[(SQLite)]
  E --> Q[轨迹]
  Q --> V[独立 Verifier]
  Q --> F[SFT / GRPO]
  F --> P
~~~

## 模块

| 位置 | 职责 |
| --- | --- |
| `agent.py` | 决策循环、LocalModelPolicy、确定性测试策略 |
| `environment.py` | Action 执行、预算、错误反馈、Check/Finish |
| `tools.py`、`sql_worker.py` | Schema、业务字典、受限只读 SQL |
| `rewards.py` | 独立校验最终答案和业务步骤 |
| `trajectories.py`、`evaluation.py` | 保存、回放、评测与指标 |
| `sft_training.py`、`qwen3_sft.py`、`lora.py` | 监督数据、mask、LoRA、checkpoint |
| `grpo/` | 分组采样、优势、token loss、更新和恢复 |
| `model_server.py`、`serving.py` | 模型 HTTP 服务、任务 API |
| `service_config.py`、`service_lifecycle.py` | 模型身份、进程、deadline、退出清理 |

## 一次任务

模型输出六种 Action：`plan`、`schema`、`dictionary`、`sql`、`check`、`finish`。例如：

~~~json
{"kind":"sql","arguments":{"query":"SELECT name FROM customers LIMIT 5"}}
~~~

Runtime 执行 Action，把工具结果或错误作为下一轮输入。`check` 绑定已有 SQL 结果；`finish` 引用通过检查的证据，最终行列由环境提取。

Action 数、工具数、单次查询和整个 episode 都有限制。SQL 错误、无效输出和预算耗尽保留在轨迹中。

## 数据与评分

模型只看到公开任务、工具反馈和自己的历史输出。参考 SQL 与业务约束留在 Verifier 侧。训练 SQL 来自 train 轨迹中实际执行的 assistant Action。

SQL worker 使用只读 URI、SQLite authorizer、独立进程及资源限制。业务库不接受写入、ATTACH、扩展加载或多语句请求。数据库文件哈希用于回放，逻辑指纹用于核对内容。

业务评分检查最终行列、必要步骤和预算。Top-3 任务即使排名正确，缺少指定的订单数查询也会失败，见 [结果示例](results.md)。

## 训练与加载

SFT 将“已有上下文 → 当前 Action”作为监督，只对当前 assistant 输出计算 loss。Qwen3 固定配方为 54 条决策、27 次 LoRA 更新。

GRPO 按同任务的多条 episode 计算相对优势，保存原生 token IDs、old logprob 和 response mask。工具反馈属于上下文，不进入生成 token 的 loss。

加载顺序：

| 配置 | 顺序 |
| --- | --- |
| Base | fresh Base |
| SFT | fresh Base → SFT Adapter |
| GRPO | fresh Base → SFT parent → GRPO delta |

Qwen3 当前是固定 batch 两次更新；历史 1.5B 另有在线采样入口。细节见 [SFT](qwen3-post-training.md) 和 [GRPO](grpo-training.md)。

## 服务与产物

模型通过 loopback HTTP 接入 LocalModelPolicy。任务 API 负责提交、状态和轨迹查询；现成的 `sft-n` profile 绑定 1.5B SFT，Qwen3 已用于 HTTP Runtime 评测。

每个 run 保存配置、模型身份、请求/响应、工具记录、轨迹、评分和指标。没有完成技术评分的任务记为 unknown，业务失败正常进入分母。
