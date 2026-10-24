# 路线图与阶段状态

更新：2026-10-24。当前工作主线是“已具有限真实后训练/冻结评测/本地 API 证据的小规模工程”向“可复核的技术手册和运行入口”收口。T-029 只校准文档与复现说明，不新增训练、评测、部署或研究目标。

## 状态定义

| 状态 | 含义 |
| --- | --- |
| 已验证 | 指定范围有审核过的静态或运行证据；不自动扩展到更大数据、更多 seed 或生产环境 |
| 已实现 | 源码/配置/入口存在，但不等于对应能力已通过真实运行 |
| 历史计划 | 旧阶段曾计划的工作，不能覆盖后续审核结果 |
| 未来扩展 | 当前工程之外的规模化、泛化或部署目标 |

## 当前主线

| 阶段 | 当前状态 | 证据与边界 |
| --- | --- | --- |
| CPU Agent MVP | 已验证 | 10 个合成任务定义、Runtime/只读 SQL/轨迹/Reward/API；fixture 不是 LLM 能力结果 |
| M2A 推理工程 | 已实现/历史背景 | 固定 Qwen 服务、60 项冻结清单和 Slurm 入口；旧 Pending 句子保留为历史，不覆盖 E-026 |
| SFT 数据与 pilot | 已验证（限定） | train-only 数据、FP16/SDPA manual LoRA、T-017-R2 的 27 次有效更新和 N adapter；旧失败与未运行评测保留 |
| SFT N 能力面板 | 已验证（限定） | E-026/T-025：N 20/20 评分、8/20 成功；单 seed、共享 schema/template，不能推出全面可靠或跨模板泛化 |
| GRPO rollout/update | 已验证（限定后端） | 多轮 rollout、TokenRecord、response-only mask、组内优势、有限更新、恢复和 final-delta 证据；不是完整 TRL Trainer 或长训 |
| Base/N/RL 冻结对照 | 已验证（本协议） | E-026：Base 0/20、N 8/20、RL 8/20、unknown 0；N−Base +40 个百分点，RL−N 0 |
| N model/API 集成 | 已验证（两项 train smoke） | E-027/T-026：固定 N、26 HTTP/model events、两项业务结果；不是 validation/generalization |
| 服务生命周期与 CPU fixture | 已验证（CPU 合同） | T-028：25 passed，唯一 run、来源计数/null、全 tmp fixture/访问守卫；新模式未重新 V100 |
| 技术文档/复现入口 | 本轮收口 | README、架构、路线图、GRPO/服务手册和 reproducibility 与当前源码/证据对齐 |

## 当前工程真正已经交付的能力

1. 公开任务可经 Runtime 规划、schema/dictionary/只读 SQL、check/finish 形成结构化 trajectory。
2. 独立 Verifier 在模型消息之外判断 reference SQL、业务约束和最终答案；SQL 可执行性不替代业务成功。
3. train-only SFT 路径和 N adapter lineage 已有审核证据；validation/test 仍不回流训练。
4. formal GRPO 路径能保存多轮 token/logprob/mask、group advantage、预算、更新状态和 parent/delta 身份；CPU fake/tiny 与有限 V100 证据分开。
5. Frozen eval 能从 fresh Base 分别加载 Base、parent N 和 parent N + fixed final delta，并按 arm/pair 独立输出 unknown/null。
6. 固定 N 的 loopback model/API、run-global lifecycle、持久证据派生计数和全 tmp CPU fixture 已有各自范围内证据。

这些能力的代码版本应写成 branch + HEAD + dirty/untracked snapshot；fresh checkout 不包含 ignored assets，不应把源码存在写成可一键复训。

## 近期工程动作

T-029 完成后，本地后训练工程可按以下顺序做阶段总验收：

- 复核文档命令与实际 help/配置；缺少 cache、N adapter、GRPO delta、DB、benchmark plan 或 private task plan 时显式报缺。
- 以 manifest、source snapshot、run-local evidence 和 code version 对齐历史结果，不重跑旧实验，不把旧 artifact 变成当前 fresh run。
- 如需新实验，先独立冻结 task/database/model/config/seed/预算，再用新的 run 目录和 Experiment ID；不要用文档更新触发训练或完整评测。

这些是复核动作，不是本任务新增 GPU 门槛。

## 真正剩余工作与未来扩展

### 仍需工程化的复现能力

- 把 train-only 数据构建、tokenizer/mask 审计、SFT 训练和 N checkpoint lineage 整理成新的、明确依赖资产的标准 recipe；当前 T-017 pilot 脚本不能被包装为 fresh checkout 一键全量重训。
- 为每个正式入口继续维护 manifest、源文件快照、命令原文、退出码、指标分子/分母和 dirty code version。
- 对服务跨 Job allocation 的累计守卫和长期运行行为另行设计；T-028 已披露而未伪造为已实现。

### 研究扩展

- 多 seed、更多数据库/实体/业务模板和严格跨模板污染检查。
- 更长的 GRPO 训练、更多 rollout 组、稳定的 checkpoint/resume 和统计不确定性。
- 将恢复能力、SQL 错误类型、效率/成本和主业务成功率作为分开终点，而不是用 SQL 辅助统计替换业务终点。
- 对 SFT/GRPO 配方做预注册的消融或公平对照；不改变已封存的 E-026 结果。

### 部署扩展

- 外部队列、PostgreSQL 只读角色、认证、租户隔离、进程/容器隔离、指标和审计服务。
- 长期服务的多模型 registry、滚动升级、故障恢复和真实容量测试。
- 这些不是当前小规模本地后训练系统的 MVP 收口门槛。

## 不设为当前门槛的事项

- 不要求 RL 相对 N 必须正增益。E-026 的 RL−N=0 是真实结果，保留并报告，不据此反复扩训。
- 不要求当前阶段拥有生产多租户、公网部署、PG 后端、分布式队列或鉴权。
- 不把单 seed、共享模板的 N +40 个百分点写成泛化定论。
- 不把 CPU fixture/tiny/fake 结果写成 Qwen/V100 或真实业务 SQL 结果。

## 历史路线图的解释

旧 M2A/M3/M4 条目仍保留在历史文档和实施记录中，反映当时“等待真实 GPU、SFT、GRPO”的计划。当前实现和证据已经推进到 SFT N、有限 GRPO、E-026 三臂冻结评测和 E-027 API；阅读旧文档时以本页、README、各审核报告和 docs/reproducibility.md 的时间/版本边界为准。

下一阶段不是继续拆服务微修，而是完成本地后训练工程范围的总验收；若无新的实质性证据缺口，本路线图不自动启动新 Job。
