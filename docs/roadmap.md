# 下一阶段实施入口

当前验收覆盖 CPU 数据与执行链路，尚无真实 LLM 基线、SFT、GRPO 或模型性能提升数据。以下是实施顺序，不是已完成清单。

## M2：设备审计与最小兼容闭环

1. 从 `docs/environment-audit.md` 复核用户政策、个人存储 quota、可选 GPU 账户/QOS。选择单卡 gpu2Q、显式短时限，先提交已准备的 `scripts/slurm/gpu_inventory.sbatch`。不要在登录节点运行 nvidia-smi 推理/模型加载，不使用八卡长期作业。
2. 采集 GPU 型号、实际显存、驱动、CC、CUDA wheel 架构与容器执行情况，决定独立 Python 3.11/3.12 GPU 环境和 CUDA 依赖。优先 Transformers FP16 + eager，无 BF16/FA2 默认值。完整依赖锁与 CPU 锁分离。
3. 在已有许可和存储空间下下载 Qwen2.5-1.5B-Instruct 的固定 revision，先单轮 32–128 token，再使用 LocalModelPolicy 跑一个多轮任务。记录峰值 allocated/reserved、生成 token/s、首次加载和端到端 latency、错误与环境元数据。3B 待 1.5B 通过后评估。
4. 在隔离环境验证微批次 1 的 LoRA SFT 数步，FP16 GradScaler、梯度裁剪、有限 loss/grad、checkpoint 保存与恢复；保存 model/optimizer/scheduler/scaler/RNG/sampler 状态。
5. 验证上游 GRPO 后端是否能避开 FA2/BF16 路径；用真实环境按同一任务采多条不同轨迹，验证终局 reward、response-only mask、group advantage 和有限梯度。未通过前不做正式训练。
6. 产出 `docs/v100-compatibility.md`，逐条区分已运行、失败和候选。当前未生成该文件，避免把预案当作实测报告。

## M3：SFT

先把 10 个基础任务扩大到足以独立评估的业务集合：多个数据库种子、实体与业务模板；冻结 test manifest 及数据指纹。当前年份参数划分不支持泛化结论。

仅导出成功且属于 train 的轨迹，去重、验证 prompt/SQL/token 质量，转换为模型 chat template 和 assistant-only loss mask。统一记录 source task/split/database hash，拒绝 test/validation 轨迹进入 SFT。训练前冻结 Base 评测，后评测 SFT 同任务集；模型版本指向真实 checkpoint revision。

## M4：多轮 GRPO

使用固定上游版本的算法与训练框架；将本项目 env 适配到 reset/step、终局 reward 和真实多轮 rollout，而不是对单条 SQL 做离线打分后声称完成 Agent RL。

补充 rollout token IDs、logprobs、采样 seed/temperature、policy revision、环境返回的 observation masks；验证同 prompt 分组和 tool token 不参与策略损失。训练任务使用 Slurm signal/preemption 保存，明确恢复 epoch/step、optimizer/scaler/RNG 与 replay buffer 状态。先 Base/SFT/RL 三组同预算对照，再做主奖励/成本项/恢复项消融；同时报告质量和效率及多 seed 不确定性。

## M5：展示与部署

将当前 API 的固定 registry 接入实际 Base/SFT/RL 版本；补充进程外队列、部署鉴权及 PostgreSQL 只读角色。报告关联原始轨迹、运行配置、硬件与 checkpoint。BIRD 适配先核对许可与只读子集，隔离官方测试答案。

简历描述只能陈述已经实现并验证的工程能力。当前可写多轮执行、安全 SQL、可验证奖励、版本化轨迹、自动评测、FastAPI 与 Slurm CPU 验收；不能填写不存在的 GRPO 收益或 GPU 吞吐数据。
