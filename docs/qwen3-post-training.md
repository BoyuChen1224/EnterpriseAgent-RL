# Qwen3-8B 训练

模型固定为 `Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`。已验证环境是 V100 32GB、FP16、SDPA math、thinking-off。

## 环境

从仓库根目录安装独立环境：

~~~bash
python3.12 -m venv .venv-qwen3
.venv-qwen3/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r requirements/qwen3-py312-linux.lock
.venv-qwen3/bin/python -m pip install --no-build-isolation --no-deps -e .
~~~

锁定 Torch 2.5.1+cu121、Transformers 4.51.3、tokenizers 0.21.1。配置位于 `configs/model/qwen3-8b.toml`。

## SFT

输入是 54 条 train 决策记录。每条保留当前上下文和目标 Action，重新用 Qwen3 chat template 分词；仅目标 Action 及结束符参与 loss。

| 参数 | 值 |
| --- | --- |
| LoRA | rank 8、alpha 16、dropout 0 |
| targets | q/k/v/o/gate/up/down，36 层、252 modules |
| 训练 | 2 epochs、microbatch 1、累积 4 |
| 更新 | 108 microbatch、27 次成功更新 |
| 优化器 | AdamW、learning rate 1e-4、clip norm 1 |
| seed | 42 |

入口：

~~~bash
.venv-qwen3/bin/python scripts/training/run_qwen3_sft.py --help
~~~

`prepare` 构建 token/mask 与计划；`tiny` 用 CPU 小模型检查训练循环；`train` 训练；`reload` 在独立进程重载 Adapter。配置见 `configs/training/qwen3-runtime-sft.toml`。

Slurm wrapper 为 `scripts/slurm/qwen3_runtime_sft_comparison.sbatch`。用 `T032_STAGE=train|reload|compare` 选择阶段，`EARL_QWEN3_RUN` 指向新的输出目录。上一阶段完成后再执行下一阶段。

训练和重载使用 1 GPU、4 CPU、64 GB。Base/SFT compare 使用两张 GPU，提交时覆盖为 8 CPU、96 GB；每张卡各加载一臂。

## GRPO

采样入口是 `scripts/evaluation/run_qwen3_grpo_rollout.py`，配置为 `configs/evaluation/qwen3-grpo-rollout.toml`；更新入口是 `scripts/training/run_qwen3_grpo_pilot.py`，配置为 `configs/training/qwen3-grpo-update-pilot.toml`。

当前更新程序复用固定的 16 条 episode，完成两次更新后独立重载、评测。它不执行新的 rollout。参数与指标见 [GRPO](grpo-training.md)。

## 需要的资产

| 输入 | 路径 |
| --- | --- |
| 模型缓存 | `artifacts/model-cache-qwen3-8b/` |
| train 决策 | `artifacts/sft-runtime-aligned-decision-pilot/t017-r2-v1/decision/tokenized-dataset.jsonl` |
| SFT parent | `artifacts/qwen3-runtime-sft-comparison/t032-v1/training/final-adapter/` |
| 固定 rollout | `artifacts/qwen3-grpo-rollout-signal/t033-v1/` |
| 冻结验证面板 | `artifacts/qwen3-runtime-sft-comparison/t032-v1/evaluation/validation/` |

这些文件留在实验环境，源码仓库不附带。配置中的路径和哈希必须与资产匹配；输出使用新目录，避免覆盖旧实验。

训练与 rollout 只用 train，validation 用于冻结评测，test 未运行。成绩为 Base 0/20、SFT 13/20、SFT+GRPO 13/20，见 [结果](results.md)。
