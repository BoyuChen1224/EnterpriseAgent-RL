# 架构与契约

当前交付面向可验证的企业分析 Agent CPU 基础设施。组件通过显式数据契约连接，不把训练框架或模型客户端嵌入 SQL 环境。

## 模块职责

| 模块 | 职责与边界 |
|---|---|
| `common.py` | 严格 Pydantic 公共契约、TOML 配置、JSON 日志；拒绝未知字段 |
| `data.py` | 固定 seed 的五表企业数据库与公开任务/隐藏参考查询注册表 |
| `agent.py` | Policy 协议、反馈驱动 fixture、本地模型 HTTP 适配、episode 循环 |
| `environment.py` | reset(task)/step(action)、预算、工具错误反馈、检查与完成状态 |
| `tools.py` | Schema/外键发现、业务字典、受限子进程调用、审计 |
| `sql_worker.py` | 只使用 stdlib 的单查询受限进程，无模型/训练/验证答案访问接口 |
| `trajectories.py` | v1.0 轨迹校验、原子保存、JSONL 导出、按数据库哈希重放 |
| `rewards.py` | episode 结束后的独立参考执行、业务约束与辅助项分解 |
| `evaluation.py` | 数据集划分选择、批量运行、质量/效率指标、同条件版本对照 |
| `serving.py` | FastAPI、任务/模型/轨迹端点、持久状态、容量限制 |
| `cli.py` | 初始化、评测、重放、对照与 API 启动入口 |

MVP 使用明确的有界 Python 状态循环。现阶段没有分叉图或复杂工作流，因此不引入 LangGraph；未来需要图编排时只替换 Runtime，保留 Environment、Policy 和轨迹契约。PostgreSQL、训练框架、模型服务端均通过后续适配器接入，不复制整个上游仓库。

## Episode 状态与证据

`reset(PublicTask)` 清空历史、SQL 结果、检查记录与预算，返回初始 Observation。策略只获得 task_id、自然语言 prompt、可见工具历史和 seed，不获得参考查询。

每次 `step(Action)` 返回 Observation：step_index、ok/result/error、terminated/reason、剩余 action/tool 预算及 seed 元数据。reward 字段保留为 null；隐藏终局奖励由独立 Reward Service 计算后写入轨迹，避免模型在未完成时探测参考答案。

Action 有六种：

1. `plan`：记录非空计划，不消耗工具次数，计入总 action 上限。
2. `schema`：发现业务表、字段类型、主键和外键。
3. `dictionary`：查询业务定义及收入口径。
4. `sql`：执行受限查询；执行错误返回 Observation，允许下轮修复。
5. `check`：检查已有 SQL 的列名和结果是否完整，不判断隐藏答案对错。
6. `finish`：必须引用通过 check 的 SQL step，最终结构化列与行由环境直接提取，模型不能在 finish 中伪造 rows。

正常终止为 completed；预算耗尽、episode 超时、策略或执行异常均保留明确原因。总 action 预算防止通过无限 plan 绕过工具次数限制。SQL 具有独立硬超时，并受 episode 剩余时间约束；可见服务协议以有超时的 HTTP 接入本地模型。受信任 Python Policy 必须遵守 decide 的 timeout 参数；当前 Runtime 不对任意第三方 Python 插件提供进程级抢占保证。

## SQL 边界

业务数据库单独存放，不包含任务答案、参考 SQL、任务状态或模型配置。Schema 工具只遍历固定五个业务表，模型不能提供 PRAGMA 字符串。SQLite 查询和 Schema 发现均在新 Python 子进程中执行，不执行任意 Python；父进程超时后由 subprocess 终止并回收子进程。

子进程启用：

- `mode=ro`、`query_only=ON`、`trusted_schema=OFF`、关闭 extension loading。
- SQLite authorizer 仅放行 SELECT、递归查询、五个 main 业务表的 READ 和必要聚合/字符串函数；其余操作默认拒绝。SQLite 对 COUNT/覆盖索引的空列名回调单独处理。
- 禁止 ATTACH、DDL/DML、事务/PRAGMA、系统表、文件读取函数、randomblob 和扩展加载。
- 256 MB 地址空间、CPU 秒数限制、禁止文件增长，临时数据只在内存。
- 默认 2 秒墙钟超时；SQL 16,000 字符、64 列、100 行、64 KiB 结果上限。超量行显式 truncated，超字节结果拒绝，不静默当完整答案使用。
- SQL 审计保存哈希、task_id、step_index、成功/错误和耗时；完整 SQL 只在对应轨迹内，输出文件不提交 Git。

边界假设：数据库由受信任生成器创建，策略输出为 JSON/SQL，服务仅本地展示。SQLite authorizer 与 rlimit 不是对任意恶意数据库文件或本机进程的 OS 安全沙箱。生产升级应采用 PostgreSQL 最小权限角色、事务只读、statement_timeout、服务器资源限制和部署隔离；不要把同用户任意文件写权限暴露给策略。

## 轨迹与重放

每个 episode 保存 schema_version、run_id、PublicTask、policy version/kind、数据库 SHA-256、完整 Settings、初始 Observation、逐轮前置 Observation/Action/tool arguments/result、token/时间、最终答案、终止原因和 RewardResult。工具成本可从步记录统计；JSONL 是后续训练数据入口，当前未实现 SFT 训练转换器。

保存前验证序号连续、Observation 链、工具名称/参数、完成答案与终局证据一致；临时文件写入/fsync 后原子替换。此校验保证结构一致性，不是数字签名，也不保证被任意重写的可信源文件真实性。

重放先验证数据库哈希，再按记录的 action 重新执行并逐步比较 Observation；不重新调用模型，不比较 wall time 和 token 值。正常及错误恢复轨迹可确定性重放。资源超时具有机器依赖性，差异会报告 mismatch；策略内部异常只可重放已执行的工具前缀，terminal_reason 是采集记录，而不是声称再次重现了模型异常。涉及 SQL 日期的任务使用固定时间范围，无 NOW/random 函数。

## 奖励与评测

隐藏验证器独立执行 reference_sql，与最终证据比较列名、整数金额、行及排序；无序任务使用带计数的行集合，保留重复行语义。Top-3 任务公开要求先确认订单数量，因此还验证最终证据之前存在与独立计数查询一致的 paid_orders 中间结果，并要求两次成功 SQL。仅重复最终查询不能满足该约束。主奖励必须同时满足最终正确、业务约束和正常完成。

默认成功 1，失败 0。可选工具成本、重复调用、错误惩罚总计最多将成功奖励降到 0.9。恢复奖励只按 episode 计一次，且辅助总值不得为正；因此增加错误再恢复不能比无错误路径获得更高奖励。不以另一个 LLM 打分，也不奖励刷调用次数。SQL 执行率、恢复状态、重复次数和预算单列为诊断分量。

评测质量指标为 Task Success Rate 和 Executable SQL Rate；效率指标为 Tool Error Rate、平均工具调用/latency/token、配置单价下的端到端费用。计数包括模型发起的工具尝试和环境拒绝，不把少调用当作更正确。未知 token 计量的真实模型失败轨迹显示 null，不伪造为零；费用不包含未采集的集群账单。

对照要求相同数据库哈希、task_id 集合、seed、split、Settings 和 policy_kind，禁止将 fixture 和 LLM 报告混合对比。当前 train/validation/test 只保证任务 ID、提示词和年份参数隔离；仍共享 schema、实体与模板，样本极小，属于功能基准，不足以证明泛化。M3 前必须扩大数据并建立跨数据库/实体/模板及哈希污染检查。

## API 与模型端点

API 单进程、loopback，容量上限 8 个待执行/执行中任务。比较请求先验证和预留整批容量，失败释放全部预留，避免出现无 worker 的 queued 作业。FastAPI BackgroundTasks 执行同步 runtime；不声称是生产分布式队列。

状态存于独立 jobs.sqlite3，轨迹原子落盘。重启后原 running/queued 标为 failed/server_restarted，便于人工重试；不宣称任务透明恢复。多进程部署、认证、多租户和外部队列尚未实现。

本地模型通过 OpenAI 兼容 `/v1/chat/completions`、严格单 JSON Action 接入。请求只携带公开 prompt 与工具历史，使用 temperature=0、单轮最多 512 新 token。端点限定为当前 Slurm 节点 loopback，防止登录节点发起推理或意外访问外部计费服务。没有对真实端点执行请求；Qwen 的加载和数值兼容属于 M2。

## 后训练边界

没有训练空类或伪实现。后续 SFT 从经独立验证成功、split=train 的轨迹转换；参考查询不进入模型消息。GRPO 使用真实环境多轮 rollout，按 task 分组采样，将终局奖励和 response-only token mask 交给经过验证的上游算法。需要额外记录 rollout token IDs/logprobs、采样配置、checkpoint revision、optimizer/scheduler/scaler/RNG 状态与训练信号恢复；这些字段目前尚未实现，不能把现有 JSON action 轨迹直接称为完整 RL 训练数据。
