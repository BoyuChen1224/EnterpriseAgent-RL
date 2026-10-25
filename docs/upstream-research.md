# 技术选型记录

2026-10-15 的上游源码调研。下表记录当时读取的固定版本，不代表这些项目的当前接口。

| 项目 | 读取版本 | 采用的思路 |
| --- | --- | --- |
| Agent-STAR | `14aa665d5d0c7256e8de4168c8ab3889d2e6658e` | 工具环境、任务结束与独立终局奖励 |
| Agent Lightning | `69f5a40b36d9bff646b8e7a5532e8b78bd41eaf9` | 请求事件、轨迹与训练数据分离 |
| verl | v0.4.1 / `8d9e350ea58c7ad4b50dd14d9dcb50577242c55f` | GRPO 组内优势与 response mask |
| BIRD-INTERACT | `451fe2c3518ee1cf908d8139e2913483bd519381` | 交互 SQL 任务及环境接口 |

## 实现取舍

业务 Runtime 使用有界 Python 循环和结构化 Action，训练侧消费轨迹。奖励直接核对数据库结果和业务步骤，不调用另一个 LLM 提取答案。

未直接安装当时的完整 verl 依赖：已读版本含 FlashAttention-2、BF16 等路径，需要额外处理 V100 兼容性。项目采用本地 GRPO adapter 与 Torch backend，相关 TRL 来源和许可证见 [NOTICE](../src/enterprise_agent_rl/grpo/NOTICE-TRL.txt)。

BIRD-INTERACT 的环境包含 PostgreSQL 和可能修改数据库的交互任务。当前项目只实现 SQLite 只读环境，尚未完成 BIRD 适配；引入时需分别核对数据条款、划分、写入隔离和评分方式。

## 固定源码

- [Agent-STAR 环境](https://github.com/WxxShirley/Agent-STAR/blob/14aa665d5d0c7256e8de4168c8ab3889d2e6658e/RL/rllm/environments/base/base_env.py)
- [Agent Lightning schema](https://github.com/microsoft/agent-lightning/blob/69f5a40b36d9bff646b8e7a5532e8b78bd41eaf9/agentlightning/schemas.py)
- [verl GRPO](https://github.com/verl-project/verl/blob/8d9e350ea58c7ad4b50dd14d9dcb50577242c55f/verl/trainer/ppo/core_algos.py)
- [BIRD SQL 环境](https://github.com/bird-bench/BIRD-Interact/blob/451fe2c3518ee1cf908d8139e2913483bd519381/bird_interact_agent/src/envs/bird_interact_env/sql_env.py)

当前架构见 [架构](architecture.md)，已完成实验见 [结果](results.md)。
