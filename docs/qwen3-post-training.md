# Qwen3-8B 后训练与评测

当前主线使用 `Qwen/Qwen3-8B`，revision 固定为 `b968826d9c46dd6066d109eabc6255188de91218`。单 V100 32GB 路径采用 FP16、SDPA math 和 `enable_thinking=false`；保留官方空 thinking 标记，不事后删除模型输出。源码和配置位于 `main`。

## 环境与入口检查

在仓库根目录创建独立 GPU 环境，不覆盖历史 1.5B 环境：

```bash
python3.12 -m venv .venv-qwen3
.venv-qwen3/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r requirements/qwen3-py312-linux.lock
.venv-qwen3/bin/python -m pip install --no-build-isolation --no-deps -e .
PYTHONPATH=src .venv-qwen3/bin/python scripts/training/run_qwen3_sft.py --help
PYTHONPATH=src .venv-qwen3/bin/python scripts/evaluation/run_qwen3_runtime_panel.py --help
PYTHONPATH=src .venv-qwen3/bin/python scripts/evaluation/run_qwen3_grpo_rollout.py --help
PYTHONPATH=src .venv-qwen3/bin/python scripts/training/run_qwen3_grpo_pilot.py --help
```

锁定环境为 Python 3.12、Torch 2.5.1+cu121、Transformers 4.51.3、tokenizers 0.21.1。上述 `--help` 仅查看参数；不加载权重、运行 SQL 或训练。

## 阶段与配置

| 阶段 | 入口与配置 | 实际行为 |
| --- | --- | --- |
| Runtime 对齐 SFT | `scripts/training/run_qwen3_sft.py`；`configs/training/qwen3-runtime-sft.toml` | `prepare` 渲染 train 监督；`tiny` 做 CPU 小模型检查；`train` 训练；`reload` 独立加载最终 Adapter |
| Base/SFT 评测 | `scripts/evaluation/run_qwen3_runtime_panel.py`；`configs/model/qwen3-8b.toml` | `--prepare` 冻结面板；各 arm 经 HTTP 模型服务运行真实 Runtime，再聚合比较 |
| 分组 rollout | `scripts/evaluation/run_qwen3_grpo_rollout.py`；`configs/evaluation/qwen3-grpo-rollout.toml` | `preflight` 核对资产；`rollout` 采样 4 task × 4 episode，保存 token、logprob、mask 和正式评分；无更新 |
| 固定批次 GRPO | `scripts/training/run_qwen3_grpo_pilot.py`；`configs/training/qwen3-grpo-update-pilot.toml` | `prepare`/`tiny` 检查；`train` 两次更新；`reload` 独立重载；`compare` 新鲜 SFT/RL 配对；`finalize` 离线收尾 |

SFT 固定使用 54 条逐决策记录、2 epochs、108 个 microbatch 和 27 次成功更新；监督只覆盖当前 assistant Action。LoRA 为 rank 8、alpha 16，覆盖 36 层的 q/k/v/o/gate/up/down 共 252 个 Linear。

GRPO pilot 复用冻结的 16 条 episode、118 条模型请求记录，70 条非零优势记录参与 backward，完整 token 分母为 4745。两次更新复用同一批数据；该入口不执行新的 rollout，不代表多轮在线训练。

GPU 阶段必须在兼容计算节点上运行。现有 wrapper 为：

- `scripts/slurm/qwen3_runtime_sft_comparison.sbatch`：`T032_STAGE=train|reload|compare`，`EARL_QWEN3_RUN` 指定 run。训练/重载使用单 GPU，Base/SFT compare 需要两张 GPU、8 CPU、96G，并覆盖 wrapper 的单卡默认资源。
- `scripts/slurm/qwen3_grpo_rollout.sbatch`：Qwen3 分组 rollout。
- `scripts/slurm/qwen3_grpo_update_pilot.sbatch`：`T034_STAGE=train|reload|compare`，`EARL_T034_RUN` 指定 run，`EARL_T034_CONFIG` 指定配置；每阶段单 GPU。

wrapper 含集群账户、QoS、分区和历史路径默认值；迁移机器时先逐项核对。新实验使用独立 run，固定配置中的来源与 hash 也须匹配，不能只换输出目录就假定历史数据已存在。

## 固定资产与加载顺序

| 资产 | 配置绑定的本地来源 |
| --- | --- |
| Qwen3 tokenizer/weights/cache manifest | `artifacts/model-cache-qwen3-8b/` |
| 54 条 train 决策记录 | `artifacts/sft-runtime-aligned-decision-pilot/t017-r2-v1/decision/tokenized-dataset.jsonl` |
| SFT parent | `artifacts/qwen3-runtime-sft-comparison/t032-v1/training/final-adapter/` |
| GRPO token/组奖励来源 | `artifacts/qwen3-grpo-rollout-signal/t033-v1/` |
| 冻结 validation 面板/数据库/私有评分输入 | `artifacts/qwen3-runtime-sft-comparison/t032-v1/evaluation/validation/` |

这些资产不随源码提交。fresh checkout 可安装依赖和查看入口；重现实验还需要取得匹配的固定资产。默认不重建旧 run、不替换 hash 或从 validation/test 生成训练监督。

SFT parent 权重 SHA-256 为 `413d1b0211ba93b2df322dcdaeb1fa1c3f019210c8ded510a2920e5ad6e114e9`。本轮 GRPO delta 为 `dc46efecbd519142ccca44fa3291861177f299bf0e599cced65094fea9e1260a`，保留在 `artifacts/qwen3-grpo-update-pilot/t034-v1/`。RL 必须加载 **fresh Base → merge parent SFT → merge GRPO delta**，delta 不能脱离 parent 单独使用。

## 已完成结果与数据边界

Base/SFT 配对为 0/20 与 13/20；后续 fresh SFT/RL 配对均为 13/20。每臂均 20/20 scored、unknown=0，RL 无 gain/loss。结果汇总见 [README](../README.md)。

训练与 rollout 只访问 train；validation 仅用于冻结评测，test 保持封存。reference SQL/expected answer 由独立 Verifier 使用，不进入业务模型请求。SQL 可执行、训练 loss、组内奖励方差和最终业务成功分开报告。

Qwen3 已通过真实 HTTP 模型服务与 Runtime 评测；`configs/serving/sft-n.toml` 仍绑定历史 Qwen2.5-1.5B N，不能据此声称 Qwen3 的同名 Agent API profile 已完成长期服务验证。
