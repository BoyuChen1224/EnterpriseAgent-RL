# EnterpriseAgent-RL

企业数据分析 Agent 后训练系统，覆盖 **多轮工具调用 → SQL 环境反馈 → 轨迹采集 → SFT → GRPO → 评测 → 模型 API**。

## 实现内容

- **Agent Runtime**：规划、Schema/Dictionary 查询、只读 SQL、结果检查和有界多轮交互。
- **数据与评测**：版本化轨迹、train-only 监督数据、独立业务 Verifier，以及 Base/SFT/RL 配对评测。
- **后训练**：LoRA SFT、真实环境 GRPO rollout、token/logprob/mask 对齐、策略更新和 checkpoint 恢复。
- **模型服务**：FastAPI 任务提交、状态与轨迹查询；固定 SFT 模型身份、进程生命周期和退出清理。

已验证模型为 **Qwen2.5-1.5B-Instruct**，单 V100 使用 FP16 + SDPA math。

## 实验结果

同一组 20 个 validation 任务、相同冻结评测协议：

| 模型 | 业务成功 | 成功率 |
| --- | ---: | ---: |
| Base | 0/20 | 0% |
| SFT N | 8/20 | 40% |
| SFT N + GRPO delta | 8/20 | 40% |

SFT 相对 Base 提升 **40 个百分点**；本轮有限 GRPO 未带来额外成功率收益。结果来自单 seed、小规模共享 schema/template 面板，尚不能说明跨库泛化。真实 N 模型的 HTTP → Agent → SQL → Verifier 服务链路也已在 V100 验证。

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
