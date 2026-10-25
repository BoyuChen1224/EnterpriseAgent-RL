# 路线图与阶段状态

更新：2026-10-25。当前主线已推进到 Qwen3-8B：Runtime 对齐 SFT 在固定面板上从 0/20 提升到 13/20；固定批次 GRPO 两更新、独立重载和完整 SFT/RL 对照通过，两臂仍均为 13/20。历史 1.5B 后训练与本地 API 作为已完成工程基础保留。当前源码统一维护在 `main`。

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
| Qwen3 底座与 LoRA | 已验证（限定） | T-031：V100 32GB、FP16/SDPA math、thinking-off、真实更新与重载；train-only 诊断不混为 validation |
| Qwen3 Runtime 对齐 SFT | 已验证（限定） | T-032：54 条 train 决策、108 微批、27 次更新；Base/SFT 各 20/20 scored，0/20 对 13/20，unknown=0 |
| Qwen3 分组 rollout | 已验证（奖励信号） | T-033：16/16 episode 正式评分，2/4 组有奖励方差；token/logprob/mask 绑定通过，无更新 |
| Qwen3 固定批次 GRPO | 已验证（有限更新/重载） | T-034：118 请求记录、70 非零优势、140 backward、2 次更新；fresh Base→parent SFT→delta 重载通过 |
| Qwen3 SFT/RL 冻结对照 | 已验证（完整零收益） | T-034：两臂各 20/20 scored、unknown=0，均 13/20，无 gain/loss；没有新在线 rollout 或长训收益结论 |
| 历史 1.5B 后训练/恢复 | 已验证（限定后端） | E-026：Base 0/20、N 8/20、RL 8/20；历史真实在线更新和确定恢复证据不外推为 Qwen3 GPU 恢复等价 |
| 历史 N model/API 集成 | 已验证（两项 train smoke） | E-027/T-026：1.5B N、26 HTTP/model events；Qwen3 已走 HTTP Runtime 评测，同名 Agent API profile 仍未升级验证 |
| 服务生命周期与 CPU fixture | 已验证（CPU 合同） | T-028：25 passed，唯一 run、来源计数/null、全 tmp fixture/访问守卫；新模式未重新 V100 |
| 技术文档/复现入口 | 已更新 | README、Qwen3 指南、架构、GRPO 和复现说明区分当前 8B 与历史 1.5B；固定资产仍在本地 |

## 当前工程真正已经交付的能力

1. 公开任务可经 Runtime 规划、schema/dictionary/只读 SQL、check/finish 形成结构化 trajectory。
2. 独立 Verifier 在模型消息之外判断 reference SQL、业务约束和最终答案；SQL 可执行性不替代业务成功。
3. Qwen3 train-only Runtime 对齐 SFT 的固定配方、真实训练、Adapter lineage 与独立重载已验证；validation/test 不回流训练。
4. formal GRPO 路径能保存多轮 token/logprob/mask、group advantage、预算、更新状态和 parent/delta 身份；CPU fake/tiny 与有限 V100 证据分开。
5. Frozen eval 能从 fresh Base 分别加载 Base、parent N 和 parent N + fixed final delta，并按 arm/pair 独立输出 unknown/null。
6. 固定 N 的 loopback model/API、run-global lifecycle、持久证据派生计数和全 tmp CPU fixture 已有各自范围内证据。

这些能力的代码版本应写成 branch + HEAD + dirty/untracked snapshot；fresh checkout 不包含 ignored assets，不应把源码存在写成可一键复训。

## 近期工程动作

当前工程维护重点：

- 源码统一在 `main`，按实际阶段拆分 commit 并同步 GitHub；历史阶段由提交历史保留，不依赖长期堆积开发分支。
- 文档命令与实际 help/配置一致；缺少 cache、SFT parent、GRPO delta、DB 或冻结 task plan 时显式报缺。
- 以 manifest、source snapshot、run-local evidence 和 code version 对齐历史结果，不重跑旧实验，不把旧 artifact 变成当前 fresh run。
- 如需新实验，先独立冻结 task/database/model/config/seed/预算，再用新的 run 目录和 Experiment ID；不要用文档更新触发训练或完整评测。

文档与 Git 整理不触发新的 GPU 任务。

## 真正剩余工作与未来扩展

### 仍需工程化的复现能力

- Qwen3 已有固定 54 条监督的训练与重载入口；继续完善从公开来源构建监督、资产取得和全流程迁移说明。固定 ignored 轨迹及数据库未随仓库发布，fresh checkout 尚不能一键复现实验全链。
- 为每个正式入口继续维护 manifest、源文件快照、命令原文、退出码、指标分子/分母和 dirty code version。
- 对服务跨 Job allocation 的累计守卫和长期运行行为另行设计；T-028 已披露而未伪造为已实现。

### 研究扩展

- 多 seed、更多数据库/实体/业务模板和严格跨模板污染检查。
- GRPO 有效性研究：先量化更新前后轨迹概率、优化方向和合并效果，再依据证据设计更多 train 任务或新鲜在线 rollout；不因当前零收益机械扩训或扫参。
- 将恢复能力、SQL 错误类型、效率/成本和主业务成功率作为分开终点，而不是用 SQL 辅助统计替换业务终点。
- 对 SFT/GRPO 配方做预注册的消融或公平对照；不改变已封存的 E-026 结果。

### 部署扩展

- 外部队列、PostgreSQL 只读角色、认证、租户隔离、进程/容器隔离、指标和审计服务。
- 长期服务的多模型 registry、滚动升级、故障恢复和真实容量测试。
- 这些不是当前小规模本地后训练系统的 MVP 收口门槛。

## 不设为当前门槛的事项

- 有限工程验收与 RL 业务增益分开。1.5B 和当前 Qwen3 pilot 的 RL 增益均未观察到，保留零收益，不将更新成功描述成能力改善。
- 不要求当前阶段拥有生产多租户、公网部署、PG 后端、分布式队列或鉴权。
- 不把单 seed、共享模板的 SFT +65 个百分点或历史 N +40 个百分点写成泛化定论。
- 不把 CPU fixture/tiny/fake 结果写成 Qwen/V100 或真实业务 SQL 结果。

## 历史路线图的解释

旧 M2A/M3/M4 和 E-026/E-027 条目保留对应历史阶段；最新 Qwen3 主线为 T-031 接入、T-032 SFT、T-033 分组采样、T-034 固定批次更新与对照。阅读时以模型身份、代码版本和评测协议区分，详见 [Qwen3 指南](qwen3-post-training.md) 与 [复现说明](reproducibility.md)。

当前固定配方与批次已收口。后续研究或 Qwen3 服务集成按新的明确任务推进，本路线图不自动启动新 Job。
