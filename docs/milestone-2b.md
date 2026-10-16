# Milestone 2B：真实推理验收与 Base Baseline

检查日期：2026-10-16（Asia/Shanghai）。本阶段从
`dev/qwen-base-inference@6a4155b` 创建 `dev/gpu-baseline`。收费 GPU 作业尚未获得用户明确授权，因此本阶段只完成缓存、静态检查、CPU 回归、评测边界和训练数据准备审计；任何 CPU fixture、tokenizer 检查或 HTTP 测试替身都不计为 Qwen 推理结果。

## 当前状态

| 项目 | 状态 | 证据/边界 |
|---|---|---|
| Git 分支 | 已完成 | `dev/gpu-baseline`，从 `6a4155b` 创建 |
| 固定模型 | 已完成 | `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| 权重缓存 | 已完成 | `experiments/m2b-preparation/qwen-cache-verification.json`；`model.safetensors` 3,087,467,144 bytes，SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| 缓存完整性 | 已完成 | 8 个 snapshot 文件逐文件 SHA-256，`incomplete_blobs=[]`，唯一物理缓存大小 3,098,967,011 bytes |
| GPU 作业 | Pending | 未提交；用户尚未授权收费 GPU |
| GPU 设备/驱动/显存/真实生成 | Pending | 必须由计算节点作业产生，不能由登录节点或 CPU 结果替代 |
| CPU 回归 | 已实测 | 本分支改动后 `.venv/bin/python -m pytest -q` 为 69 passed |
| SFT / GRPO | 未执行 | 本阶段仅审计训练数据，不进行正式训练 |

缓存下载复用了既有 Hugging Face cache 的 `.incomplete` blob 并完成续传；随后以 `HF_HUB_OFFLINE=1` 重建 manifest 和独立校验。模型服务要求 manifest、固定 revision 和 `local_files_only=True`，GPU 作业不会联网下载。

## 三层验收

`scripts/inference/run_qwen.py` 现在将结果分为：

1. Infrastructure Smoke：节点上的模型加载、loopback health、非空文本生成、HTTP 服务和子进程回收。对应 `single` 与 `scripts/slurm/qwen_single.sbatch`。
2. Protocol Smoke：真实模型生成协议有效 JSON Action，并驱动真实 SQLite 工具；不以业务成功替代协议结论。对应 `protocol` 与 `scripts/slurm/qwen_protocol.sbatch`。
3. Agent Capability Smoke：真实模型运行 `top-2025` 多轮任务，由独立 verifier 打分。业务失败会保留完整 trajectory、reward 和错误原因，标记为 `business_failed`，不会被报告成基础设施失败。对应 `agent` 与 `scripts/slurm/qwen_agent.sbatch`。

Base validation 默认只跑 `validation` split；`test` 仅在模型 revision、prompt、工具协议、生成参数、reward 和 runtime 配置冻结后正式运行。`scripts/slurm/qwen_base.sbatch` 的验证器异常会使报告 `complete=false`，但不会删除该任务的轨迹。

## 预提交检查

不需要 GPU 的检查命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-gpu/bin/python scripts/setup/verify_qwen_cache.py
.venv-gpu/bin/python scripts/setup/freeze_m2b.py
for f in scripts/slurm/*; do bash -n "$f"; done
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

`sbatch --test-only` 只检查调度参数，不启动 GPU 作业；如集群允许，将明确提供账户、QOS、单卡数量和时限，并把原始输出写入准备记录。没有实际 `sbatch` 提交时，作业 ID 必须记录为 `Pending`。

授权后的顺序固定为：

```text
gpu_inventory → single/infrastructure → protocol → agent → validation → frozen test
```

每个阶段使用独立输出目录和原始日志；不因单轮加载成功而宣称 Agent 验收通过。需要提交时，先报告账户、QOS、1 张 GPU、作业时限和预计资源开销，再由用户决定。

## Base 报告口径

Base report 至少记录任务完成率、验证器成功率、Action JSON 有效率、SQL 可执行率、无效工具调用、平均工具调用、平均输出 token、生成延迟、Episode 延迟、GPU 峰值显存、错误类型分布和每个失败任务的 trajectory 文件。业务失败进入 scheduled denominator；验证器异常单独记录并使评测不完整。`failure_trajectories` 中的路径指向可重放的完整 JSON trajectory。

模型上下文只接收 public task、系统协议和真实工具反馈，不接收 reference SQL、oracle 或测试答案。生成 attempts 会保留原始文本、重试、token usage、服务端 telemetry 和错误。

## 训练数据审计与扩展

`scripts/evaluation/audit_train_data.py` 只使用 `split=train`，通过真实工具执行并由独立 verifier 检查 deterministic expert 轨迹，输出
`experiments/m2b-preparation/train-data-audit.json`。这些是标记为
`deterministic_fixture_expert` 的专家示范，不是 Base 能力结果，也不应与 `local_llm` 轨迹混合。

当前 train-only 审计实际调度 20 条任务，独立 verifier 成功评分 20 条，但 deterministic expert 只有 6 条成功、14 条失败（12 条策略覆盖/运行异常，2 条业务结果错误），10 个模板、2 个数据库、3 种动作序列；详见 `train-data-audit.json`。这证明 20 条任务连现有专家覆盖都不足，不能支撑强泛化结论。后续应扩大数据库实例、schema/行业口径、时间和实体分布、任务模板、错误恢复和负例；同时收集真实 Base 的成功与失败轨迹，保留 invalid JSON、工具错误、SQL 修复、超时和验证失败，严格 train-only 生成 SFT 数据，validation 只用于选择，test 只用于最终报告。GRPO 还需真实 response token/logprob、response mask、多样 rollout 和可恢复状态，本阶段不实现。

## 原始证据目录

- 缓存 manifest：`artifacts/model-cache/qwen-cache-manifest.json`（本地忽略，不提交权重）。
- 缓存校验：`experiments/m2b-preparation/qwen-cache-verification.json`。
- 配置冻结：`experiments/m2b-preparation/experiment-freeze.json`。
- train 数据审计：`experiments/m2b-preparation/train-data-audit.json`。
- GPU 运行后：`artifacts/gpu-inventory-<job>/`、`artifacts/qwen-<mode>-<job>/`，包括 execution、scheduler、nvidia-smi、model-load、generations、原始日志和 trajectories。
