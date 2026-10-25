# 当前架构与数据契约

EnterpriseAgent-RL 由可验证业务 Runtime、模型 Policy、只读 SQL 工具、独立 Verifier、轨迹证据、后训练入口、冻结评测和本地服务组成。业务 Runtime Agent 位于 src/enterprise_agent_rl/agent.py；项目统筹者和总控文档不是 Runtime 的一部分。

## 端到端关系

~~~mermaid
flowchart LR
  P[PublicTask] --> R[Runtime]
  R --> M[Policy]
  M -->|business messages| R
  R --> E[Environment]
  E --> T[Schema / Dictionary / SQL tools]
  T --> W[read-only SQL worker]
  W --> D[(business SQLite)]
  E --> Q[Trajectory]
  Q --> V[independent Verifier]
  V --> Z[Reward / evaluation evidence]
  Q --> S[SFT train-only rows]
  S --> L[SFT adapter]
  L --> G[GRPO rollout + update]
  G --> Z
  L --> H[model_server + Agent API]
~~~

同一条链路中的“模型消息”和“评分证据”是两个边界：模型只能看到公开任务、环境 observation、工具返回值和自己的历史输出；reference SQL、expected answer、业务约束、Verifier payload 和最终 RewardResult 由私有评分路径保留。

## 模块职责

| 模块 | 当前职责 |
| --- | --- |
| common.py / data.py | Pydantic/TOML 契约、日志、固定 seed 的合成业务数据库、公开任务和 Verifier 注册表 |
| agent.py | Policy 协议、确定性 fixture Policy、LocalModelPolicy、单 episode 决策循环 |
| environment.py | reset/step、action/tool 预算、错误反馈、完成条件和最终结构化答案 |
| tools.py / sql_worker.py | schema、业务 dictionary、只读 SQL 子进程、审计和资源上限 |
| trajectories.py | v1.x 轨迹校验、原子保存、JSONL、数据库 hash 绑定和 replay |
| rewards.py | 独立 reference SQL/业务约束检查；不把 SQL 可执行性当业务成功 |
| evaluation.py | split 选择、批量运行、Task Success/Executable SQL/效率指标和同条件对照 |
| serving.py | FastAPI loopback API、任务状态、持久 jobs/trajectory、模型 registry 和本地比较 |
| service_config.py / service_lifecycle.py | SFT N profile 身份校验、run 分配、deadline、信号收尾、持久运行证据派生计数 |
| model_server.py | 单卡 Transformers Qwen 服务；启动时才加载 tokenizer/model/cache/adapter |
| grpo/ | formal rollout、TokenRecord、EvidenceStore、预算、LoRA 更新、冻结 Base/N/RL 评测和聚合 |
| scripts/training/run_sft.py | train-only 手工 LoRA SFT pilot 入口；需要 Slurm，不能据脚本名推断 current N 已可 fresh 重训 |
| scripts/training/run_grpo.py | formal GRPO wrapper；显式 repo/config/run，分出 CPU fake/tiny、real train、eval-only、offline closeout |
| qwen3_sft.py / scripts/training/run_qwen3_sft.py | Qwen3 固定 54 条 Runtime 决策监督、assistant-only mask、27 次 LoRA 更新与独立重载 |
| grpo/qwen3_rollout.py / grpo/qwen3_update.py | Qwen3 原生 token/logprob/mask 契约、冻结 rollout 批次与父依赖 GRPO delta |
| scripts/training/run_qwen3_grpo_pilot.py | Qwen3 固定批次两更新、独立重载和 fresh SFT/RL 冻结比较；没有在线采样阶段 |
| scripts/serving/run_sft_n_api.py | 模型子进程 + Agent API 子进程的 bounded smoke/serve supervisor |

当前实现是有界 Python 状态循环和显式文件证据，不是隐式工作流服务。

## Runtime、Policy 与环境

一次 episode 的输入是 PublicTask。Runtime 给 Policy 的可见信息包括 task_id、公开 prompt、observation、已返回的 schema/dictionary/SQL 结果、错误和历史 assistant 输出；Policy 不读取 reference SQL 或 expected answer。

可用 Action 为：

1. plan：记录计划，占 action 预算，不执行工具。
2. schema：发现业务表、字段类型、主键和外键。
3. dictionary：查询公开业务定义和口径。
4. sql：把 SQL 交给只读 worker；错误作为下一轮 observation 返回。
5. check：检查列名/结果结构和已执行 SQL 的证据，不判断隐藏答案。
6. finish：引用通过 check 的 SQL step，最终行列由环境提取，不能由模型伪造。

环境保留 action/tool budget、episode timeout、每次 SQL timeout 和 terminal reason。Policy 的格式错误、SQL 错误、预算耗尽和业务错误都保留在 trajectory；“模型返回了 JSON”不等于任务成功。

## 只读 SQL 与独立评分

SQL worker 运行在新 Python 子进程中，使用 read-only SQLite URI、query_only、trusted_schema 关闭、extension loading 关闭、authorizer 白名单和 rlimit。禁止 DDL/DML、ATTACH、PRAGMA、文件函数和扩展加载；默认限制 2 秒、256 MB、100 行和 64 KiB 结果。被截断或拒绝的结果不能成为完成证据。

业务数据库只保存企业业务表，不向模型暴露 reference SQL、任务答案或 Verifier 状态。独立 Reward/Verifier 在 episode 终止后执行隐藏 reference SQL、业务约束和最终答案比较；SQL 执行成功、工具调用数和 Verifier success 是不同字段。

当前服务仍假设受控的本地 SQLite 与可信生成器。生产数据库需要 PostgreSQL 最小权限角色、statement timeout、部署隔离和鉴权；这些不是当前实现。

## 模型消息与私有评分字段

| 类别 | 内容 | 是否进入模型 messages |
| --- | --- | --- |
| PublicTask | task_id、公开 prompt、公开 split 元数据 | 是 |
| Observation | schema/dictionary/SQL 工具结果、错误、预算和终止提示 | 是，按 Runtime 可见历史返回 |
| Assistant output | 原始模型输出、解析后的 Action、格式重试消息 | 是，后续轮次可见；原始输出另存 |
| Token evidence | prompt/generated token IDs、old/raw/behavior log-probabilities、response mask、request/response IDs | 训练证据；不是业务 prompt 的隐藏字段 |
| Private verifier | reference SQL、expected answer、隐藏业务约束、verifier_payload | 否，只给隔离 Verifier |
| Scoring result | RewardResult、success、reason、Verifier SQL count、technical unknown | 否，episode 后由评分路径写入 |
| Provenance | model identity、adapter/delta hash、数据库 hash、event/trajectory/call-ledger、source kind | 否；用于审计、聚合和复现 |

因此，模型能看到工具返回的业务结果，但看不到“这个结果是否等于 reference answer”的评分标签。

## 轨迹与持久证据

每条 trajectory 绑定 schema version、run_id、PublicTask、policy identity、数据库 hash、settings、逐步 observation/action/tool result、raw model attempts、token usage、terminal reason 和 RewardResult。原子写入和 schema 校验保证结构链连续；它们不是数字签名，也不证明任意外部文件可信。

formal GRPO 还为每次模型尝试先写不可变 INTENT，再绑定 response、engine event、TokenRecord、Action attempt 和 trajectory。TokenRecord 要求：

- response_mask 的 prompt token 为 0、generated response token 为 1；
- generated token、raw policy log-probability、behavior sampling log-probability 长度相等且有限；
- request/response/event/path hash 绑定，不用进程内计数器覆盖旧证据；
- 缺响应、hash 冲突或不完整链在送 Verifier 前变成 technical unknown。

服务 summary/checks 的 model_calls、business_sql_calls、verifier_calls 只从有来源的持久 model event、SQL audit 或最终 trajectory 派生；主来源与 fallback 互斥。没有来源就写 null 和 reason，不把缺测写成 0；gpu_calls 与 scheduler allocation 也不凭空补零。

## SFT：train-only 轨迹到 Adapter

SFT 监督输入只允许来自获准 train split 的实际执行 assistant Action/轨迹；validation/test 和 reference SQL 不回流模型消息。SFT 数据需保留 source task、split、数据库指纹、trajectory source 和 token/mask 摘要。

scripts/training/run_sft.py 是当前仓库中的受控单卡手工 LoRA pilot：读取 dataset JSONL 与 metadata，要求 SLURM_JOB_ID，使用固定 Qwen revision、FP16、SDPA 和 manual LoRA，并写 checkpoint/optimizer/scheduler/scaler/RNG。它不是一个隐藏下载、自动准备数据库、自动生成 current N 的全量复训器。

当前 Qwen3 SFT 入口为 `scripts/training/run_qwen3_sft.py`，配置为 `configs/training/qwen3-runtime-sft.toml`。54 条 train 决策经当前 tokenizer 重渲染，2 epochs、108 microbatch、27 次成功更新；LoRA 覆盖 36 层 × 7 targets = 252 modules。Base 冻结，FP16 AMP 与 SDPA math 覆盖 forward、backward 和 checkpoint recompute。冻结 parent 权重 SHA-256 为 `413d1b0211ba93b2df322dcdaeb1fa1c3f019210c8ded510a2920e5ad6e114e9`。

历史 1.5B N 的 lineage 是 T-017-R2 的 final-step-27 adapter；weight SHA-256 为 e67372df6d805d4ece08e6e698ccbe76d73cc8f83fe96b12f6c7939028f088b9，adapter config SHA-256 为 bae7f0af3379a71ae24b0386131d1728eea4ed48fef8c7a0f3ba22780ce17c22。历史训练日志和源码快照保留在对应的本地 artifacts 目录；源码提交不包含这些过程记录。

## GRPO：多轮 rollout、token loss 与更新

历史 1.5B formal GRPO 的在线 train 路径由 run_grpo.py train 调用 grpo.orchestration.run_online_training：

1. 按 task/group 采样多条真实多轮 episode，并保存每个模型 request 的 messages、response、token IDs 和模型身份。
2. TokenRecord 保存 prompt/generated IDs、response-only mask、raw policy log-probabilities 和 behavior sampling log-probabilities。tool token 不被 mask 成 response loss。
3. 对同一 task group 的终局 reward 计算组内优势，使用 sample standard deviation 加 1e-4；零方差组产生零优势并跳过对应更新。
4. 按配置执行两轮、每轮两次 logical update；一个进程内保留 policy、optimizer、scheduler、scaler 和 RNG 状态，更新后刷新 policy snapshot。
5. 保存 rollout fingerprint、score/index、training state、parent stack 和 final delta；资源/证据缺失时终点为 unknown 或 null。

这里的上游算法是仓库内的 LocalGRPOAdapter 与本地 torch backend 适配，不是安装完整 TRL Trainer，也不是 vLLM、PEFT、Accelerate、分布式队列或外部训练服务。

Qwen3 使用独立的 `run_qwen3_grpo_rollout.py` 采集 4 task × 4 episode，再由 `run_qwen3_grpo_pilot.py` 复用固定批次进行两次更新。118 条模型请求中，70 条有非零优势、48 条为零；优势按 episode 的组内相对 reward 绑定到请求。loss 只包含模型生成 token，完整分母为 4745 token，零优势记录保留在分母中。两次更新共 140 次 backward；该 pilot 没有使用更新后策略重新采样，不等于历史在线入口的两轮训练。

## Base、N、RL 的加载身份

冻结评测和正式对照都要求每个 arm 从同一 fresh Base revision 开始；评测时不把上一个 arm 的内存对象继续使用。

| Arm | fresh model load | adapter merge | 解释 |
| --- | --- | --- | --- |
| Base | Qwen base 一次 | 0 | 未训练对照 |
| N | Qwen base 一次 | parent N 一次 | SFT N |
| RL | Qwen base 一次 | parent N 一次，再 fixed final delta 一次 | N 的 GRPO delta；不是独立 Base→RL 替代模型 |

当前主线身份为 `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`，V100 路径为 FP16 + SDPA math、thinking-off。Qwen3 GRPO delta SHA-256 为 `dc46efecbd519142ccca44fa3291861177f299bf0e599cced65094fea9e1260a`，必须在指定 parent SFT merge 后加载，不能作为独立 Adapter 套在原始 Base 上。

历史 Qwen2.5-1.5B-Instruct revision 为 989aa7980e4cf806f80c7fef2b1adb7bc71aa306，T-023/E-025 的 final-delta 由历史冻结评测 TOML 绑定。两条模型链使用各自的 cache、Adapter、配置与环境，不混用。

## 评测与 unknown 口径

Qwen3 固定 20 项 validation panel 上，两组配对评测分别为 Base/SFT 0/20 对 13/20，以及 fresh SFT/RL 13/20 对 13/20；每臂均 20/20 scored、unknown=0。两次更新未改变任何任务的成功/失败。它们是单 seed、小规模共享模板结果，不能推断跨库泛化；test 未运行。

E-026 的三臂评测固定 20 项 validation panel、greedy temperature 0、seed 17、相同 Runtime/Verifier 边界。每个 arm 单独要求 planned=qualified=scored=20、unknown=0；业务失败是已评分的 false，不是 unknown。

pair gate 独立计算：

- N−Base 只依赖 Base 与 N 两臂完整；
- RL−N 只依赖 N 与 RL 两臂完整；
- RL−Base 只依赖 RL 与 Base 两臂完整；
- global three-arm COMPLETE 仍要求 Base、N、RL 三臂和三个 pair 都完整。

因此，一个缺失的 RL 结果不能把已经完整的 N−Base 变成 null；反过来，global 三臂仍会保持 incomplete。成功率、SQL Action/SQL_OK、Runtime worker、Verifier 调用和技术 unknown 分开报告。E-026 当前结果为 Base 0/20、N 8/20、RL 8/20，unknown 均为 0；这只支持该冻结 panel 的有限结论。

## API、模型服务与部署边界

Agent API 是单进程 loopback FastAPI，状态和轨迹持久化到 run-local SQLite/JSON；模型服务是单卡 Transformers HTTP 端点。SFT N launcher 负责启动、ready、smoke/serve、deadline、signal 和子进程组收尾，持久化 model event、trajectory、SQL audit、summary/checks 与 allocation/wall 分离。

当前没有 LangGraph、vLLM、完整 TRL Trainer、PostgreSQL 后端、外部分布式队列、认证、多租户、常驻公网部署或跨 Job 自动资源累计。E-027 只证明单 V100 上固定 N 的两项 train HTTP smoke；T-028 的新增 smoke/serve/lifecycle 证明是 CPU 合同，不是新的 V100 或生产部署证明。

Qwen3 已通过真实 HTTP 模型服务连接 LocalModelPolicy/Runtime 的评测，但 `configs/serving/sft-n.toml` 仍是历史 1.5B profile。当前架构支持的真实加载链与已验证的具体服务 profile 分开记录，详见 [Qwen3 后训练指南](qwen3-post-training.md)。
