# 运行与复现

源码维护在 `main`。所有命令从仓库根目录执行：

~~~bash
git clone --branch main https://github.com/BoyuChen1224/EnterpriseAgent-RL.git
cd EnterpriseAgent-RL
~~~

## CPU smoke

~~~bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/cpu-py312-linux.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python scripts/evaluation/cpu_smoke.py
~~~

smoke 使用临时 SQLite、确定性策略和本地 API，检查工具、评分与轨迹回放。

## 完整 CPU CI

~~~bash
.venv/bin/python -m pip install -r requirements/cpu-training-py312-linux.lock
.venv/bin/ruff check .
.venv/bin/ruff format --check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/evaluation/cpu_smoke.py
~~~

后训练测试使用 Torch 2.5.1+cpu、Transformers 4.51.3、Safetensors 和随机小模型，不下载模型权重。GitHub CI 另设置 `CUDA_VISIBLE_DEVICES=""`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。

两项 `local_artifacts` 测试核对历史实验包，文件未提供时明确 skip。其余数据、数据库和 venv 均由测试在临时目录创建。最新通过记录见 [测试](validation.md)。

## GPU 环境

| 模型 | 环境 | 锁文件 |
| --- | --- | --- |
| Qwen3-8B | Torch 2.5.1+cu121、Transformers 4.51.3、tokenizers 0.21.1 | `requirements/qwen3-py312-linux.lock` |
| Qwen2.5-1.5B-Instruct | Torch 2.2.2+cu121、Transformers 4.46.3、tokenizers 0.20.3 | `requirements/gpu-py312-linux.lock` |

使用各自的 `.venv-qwen3` / `.venv-gpu`。已验证设备为 V100/V100S 32GB，精度 FP16、attention 为 SDPA math。[Qwen3 操作](qwen3-post-training.md) · [V100 说明](v100-compatibility.md)

## 训练和评测资产

代码与配置已提交；模型、Adapter、数据和原始运行产物保留在实验环境。

| 资产 | 当前来源 |
| --- | --- |
| Qwen3 cache | `artifacts/model-cache-qwen3-8b/` |
| 54 条 train 决策 | `artifacts/sft-runtime-aligned-decision-pilot/t017-r2-v1/decision/tokenized-dataset.jsonl` |
| Qwen3 SFT parent | `artifacts/qwen3-runtime-sft-comparison/t032-v1/training/final-adapter/` |
| Qwen3 rollout / delta | `artifacts/qwen3-grpo-rollout-signal/t033-v1/`、`artifacts/qwen3-grpo-update-pilot/t034-v1/` |
| 1.5B SFT parent | `artifacts/sft-runtime-aligned-decision-pilot/t017-r2-v1/training/checkpoints/final-step-27/adapter/` |
| 1.5B 冻结 RL | `configs/training/grpo-frozen-evaluation.toml` 绑定的 T-023 delta |
| 数据库和评分输入 | 各 run 的冻结 plan、public panel、数据库与 Verifier 输入 |

关键 Qwen3 权重 SHA-256：

~~~text
SFT parent  413d1b0211ba93b2df322dcdaeb1fa1c3f019210c8ded510a2920e5ad6e114e9
GRPO delta  dc46efecbd519142ccca44fa3291861177f299bf0e599cced65094fea9e1260a
~~~

取得资产后，核对配置中的路径、revision 和哈希，再选择新输出目录。fresh checkout 可以跑 CPU 检查；完整复训还需要上述固定输入。

## 数据隔离与记录

SFT/rollout 只用 train；validation 用于冻结评测。模型可通过工具查询当前评测库，私有参考 SQL 和答案只交给 Verifier。test 暂未做真实模型评测。

每次运行记录 commit、工作区差异、配置、模型/数据库指纹和原始请求/响应。失败日志保留，补评分和离线收尾使用新记录。未知结果保留 null，不用零填充。

[公开示例](examples/model-comparison.json) 摘录了已有模型输出和实际查询结果，便于核对 README；完整原始轨迹留在本地。
