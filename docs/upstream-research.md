# 上游源码核对与选型

核对日期：2026-10-15。先浏览用户指定的四个项目，再以只读 shallow/partial clone 放入 `/tmp/enterprise-upstream/`，通过 `git show` 读取具体文件；没有把上游代码复制或 vendoring 到本仓库。GitHub REST tree API 被限流，因此改用 Git 对象读取。下列是实际读取的提交，而非凭项目名推测接口。

| 上游 | 已读取版本 | 实际代码与发现 | 本项目决策 |
|---|---|---|---|
| Agent-STAR | `14aa665d5d0c7256e8de4168c8ab3889d2e6658e` | `RL/rllm/environments/base/base_env.py` 的 reset/step；`RL/rllm/environments/tools/travel_tool_env.py` 的工具循环、结束条件和终局 reward；`RL/rllm/rewards/travel_reward.py` 的任务约束评分 | 借鉴环境/终局奖励分离；不依赖 TravelPlanner 数据与模型格式化服务 |
| Agent Lightning | `69f5a40b36d9bff646b8e7a5532e8b78bd41eaf9`，pyproject 标为 1.0.2 | `agentlightning/schemas.py` 的 Event/ModelRequestData/RewardData/RolloutState；`client.py` 的 HTTP 边界；`verl/rollout_adapter.py` 的训练数据适配 | Runtime 产出可回放事件/轨迹，训练侧另做适配；不假设旧版 LitAgent 接口仍是当前 API |
| verl | tag `v0.4.1`，`8d9e350ea58c7ad4b50dd14d9dcb50577242c55f` | `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`；`verl/workers/fsdp_workers.py`；`requirements.txt` | 评估现有 GRPO 算法，暂不安装。历史版本也不能简单等同 V100 兼容 |
| BIRD-INTERACT | `451fe2c3518ee1cf908d8139e2913483bd519381` | `bird_interact_agent/src/envs/bird_interact_env/sql_env.py` 的 schema/column meanings/knowledge、PostgreSQL 与 test cases；README、LICENSE | 后续独立适配交互任务与只读子集，当前不下载/训练任何 BIRD 数据 |

## Agent-STAR 的奖励边界

实际 reward 文件同时使用程序化 commonsense/hard constraints 和外部模型的格式提取调用；部分上游输入解析也不适合直接放进生产环境。本项目 action 原生结构化，奖励直接比较数据库执行结果，不需要另一个 LLM 提取答案，不复制其文本解析实现。

Agent-STAR 的源码目录包含自身 rLLM/verl 代码树；本项目不复制该目录。未来选择固定上游版本，以环境/rollout/reward 适配层接入。

源码：[BaseEnv](https://github.com/WxxShirley/Agent-STAR/blob/14aa665d5d0c7256e8de4168c8ab3889d2e6658e/RL/rllm/environments/base/base_env.py)、[TravelToolEnvironment](https://github.com/WxxShirley/Agent-STAR/blob/14aa665d5d0c7256e8de4168c8ab3889d2e6658e/RL/rllm/environments/tools/travel_tool_env.py)、[travel_reward](https://github.com/WxxShirley/Agent-STAR/blob/14aa665d5d0c7256e8de4168c8ab3889d2e6658e/RL/rllm/rewards/travel_reward.py)。

## Agent Lightning 的架构边界

此次实际 checkout 中，schema 定义了 rollout 生命周期、事件、模型请求元数据与 reward；模型端点具有版本字段，verl 适配器处理 rollout 转训练数据，包括 token 对齐。它支持的范围比本项目需要的大；MVP 不引入 Gateway/Kubernetes/controller。后续仅对接训练所需的轨迹事件和 token 信息，而不叠加两套 runtime。

其当前 pyproject 要求 Python >=3.12，并含 kr8s 等依赖；不能把整个项目当成没有部署成本的轻量库安装到 CentOS 7 环境。

源码：[schemas.py](https://github.com/microsoft/agent-lightning/blob/69f5a40b36d9bff646b8e7a5532e8b78bd41eaf9/agentlightning/schemas.py)、[rollout_adapter.py](https://github.com/microsoft/agent-lightning/blob/69f5a40b36d9bff646b8e7a5532e8b78bd41eaf9/agentlightning/verl/rollout_adapter.py)、[pyproject](https://github.com/microsoft/agent-lightning/blob/69f5a40b36d9bff646b8e7a5532e8b78bd41eaf9/pyproject.toml)。

## verl 的 Volta 风险与算法边界

v0.4.1 的 GRPO outcome advantage 使用同 prompt 分组、终局 reward 和 response mask；我们应复用经过验证的训练算法，而不是自写完整分布式 RL 框架。

但该版本 `fsdp_workers.py` 中 actor/critic 配置使用 `attn_implementation='flash_attention_2'`，mixed precision 存在 BF16 默认值；requirements 还包括 flash-attn、liger-kernel 与未锁定训练依赖。**不能因为是历史 tag 就声称它支持 V100。** 需要在隔离训练环境中验证 FP16、eager attention、response mask、GradScaler 和 checkpoint 恢复，再评估是维护小范围兼容 patch、切换后端，还是采用另一经过验证的 GRPO 实现。当前没有写一个看似接通但不能训练的 Trainer 空类。

源码：[FSDP workers](https://github.com/verl-project/verl/blob/8d9e350ea58c7ad4b50dd14d9dcb50577242c55f/verl/workers/fsdp_workers.py)、[GRPO 核心算法](https://github.com/verl-project/verl/blob/8d9e350ea58c7ad4b50dd14d9dcb50577242c55f/verl/trainer/ppo/core_algos.py)、[requirements](https://github.com/verl-project/verl/blob/8d9e350ea58c7ad4b50dd14d9dcb50577242c55f/requirements.txt)。

## BIRD 数据、只读能力与许可

上游不仅是只读 Text-to-SQL，README 明确描述 CRUD、交互阶段、外部知识与可执行 test cases；部分任务会改变数据库。当前环境禁止写入，不能直接宣称完整覆盖 BIRD-INTERACT。后续需要记录只读子集选择标准，独立 PostgreSQL 实例和评测恢复机制；如研究完整 CRUD，必须放在另一个专门受控环境中。

源码 LICENSE 为 MIT；README 同时标示数据相关的 CC BY-SA 4.0。代码许可不能替代数据集具体授权，必须核对下载时的 dataset card、来源及划分要求。当前未获取数据或 ground truth；测试答案不进入训练转换器，不把 benchmark 当训练池。

来源：[SqlEnv](https://github.com/bird-bench/BIRD-Interact/blob/451fe2c3518ee1cf908d8139e2913483bd519381/bird_interact_agent/src/envs/bird_interact_env/sql_env.py)、[README](https://github.com/bird-bench/BIRD-Interact/blob/451fe2c3518ee1cf908d8139e2913483bd519381/README.md)、[LICENSE](https://github.com/bird-bench/BIRD-Interact/blob/451fe2c3518ee1cf908d8139e2913483bd519381/LICENSE)。

## 本次明确未做

没有安装上游训练框架、执行 GPU 算子、宣称验证任一 verl/rLLM/vLLM 版本、下载 benchmark 或模型权重。CPU 依赖锁与后续 GPU 训练依赖必须分开；GPU 版本决策需基于后续设备审计，而不是上游 README 的最新安装命令。
