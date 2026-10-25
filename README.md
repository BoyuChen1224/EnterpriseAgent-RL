# EnterpriseAgent-RL

企业数据分析 Agent 后训练系统，覆盖 **多轮工具调用 → SQL 环境反馈 → 轨迹采集 → SFT → GRPO → 评测 → 模型 API**。

## 实现内容

- **Agent Runtime**：规划、Schema/Dictionary 查询、只读 SQL、结果检查和有界多轮交互。
- **数据与评测**：版本化轨迹、train-only 监督数据、独立业务 Verifier，以及 Base/SFT/RL 配对评测。
- **后训练**：LoRA SFT、真实环境 GRPO rollout、token/logprob/mask 对齐、策略更新和 checkpoint 恢复。
- **模型服务**：FastAPI 任务提交、状态与轨迹查询；固定 SFT 模型身份、进程生命周期和退出清理。

当前主线模型为 **Qwen3-8B**，已在单张 V100 32GB 上验证 LoRA SFT、GRPO 更新与模型重载，使用 **FP16 + SDPA math，关闭 thinking**。保留 Qwen2.5-1.5B-Instruct 的历史实现与实验。

## 实验结果

Qwen3-8B 在固定的 20 个 validation 任务上完成 Base/SFT 与 SFT/RL 两组配对评测；每臂均 20/20 正式评分，technical unknown=0：

| 模型 | 业务成功 | 成功率 |
| --- | ---: | ---: |
| Qwen3-8B Base | 0/20 | 0% |
| Runtime 对齐 SFT | 13/20 | 65% |
| SFT + GRPO delta | 13/20 | 65% |

SFT 使用 **54 条 train-only 逐决策监督记录、27 次 LoRA 更新**，相对 Base 提升 **65 个百分点**。GRPO 从 16 条真实多轮 episode 中获得 2/4 个有奖励差异的任务组，复用固定批次完成 **2 次真实策略更新**；本轮未带来额外业务收益，20 个任务均无成功/失败变化。

结果来自单 seed、两个合成数据库实例和共享 schema/template 的小面板，test 未运行。历史 1.5B 面板为 Base 0/20、SFT 8/20、RL 8/20；两套模型实验的配方和协议不同，不将其差值归因于模型规模。模型、数据库和原始运行产物不随源码发布。

## CPU 快速开始

Python 3.12；从仓库根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/cpu-py312-linux.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
PYTHONPATH=src .venv/bin/python scripts/evaluation/cpu_smoke.py
```

该 smoke 使用临时 SQLite 和确定性 Policy，检查执行、轨迹重放与 API，不加载模型，也不代表 LLM 能力。

## 运行入口

| 功能 | 入口 |
| --- | --- |
| Agent / 评测 / API CLI | `python -m enterprise_agent_rl.cli --help` |
| SFT 训练 | `scripts/training/run_sft.py` |
| GRPO 训练 / 冻结评测 | `scripts/training/run_grpo.py` |
| SFT N 模型服务 | `scripts/serving/run_sft_n_api.py` |
| Slurm 示例 | `scripts/slurm/grpo_online.sbatch`、`scripts/slurm/sft_n_api.sbatch` |

GPU 入口需要对应的模型、Adapter、数据与配置资产，通过 Slurm 在兼容节点运行。模型、数据库、checkpoint、实验产物和协作提示词不随源码提交；当前 N 的完整标准化复训流程仍待整合。

## 技术文档

[系统架构](docs/architecture.md) · [GRPO 训练与评测](docs/grpo-training.md) · [模型服务](docs/model-serving.md) · [复现与资产说明](docs/reproducibility.md) · [路线图](docs/roadmap.md)
