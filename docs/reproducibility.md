# 可复现运行入口与资产边界

源码统一维护在 `main`；历史阶段的代码保留在 commit 历史中。获取完整当前源码：

```bash
git clone --branch main https://github.com/BoyuChen1224/EnterpriseAgent-RL.git
cd EnterpriseAgent-RL
```

当前 Qwen3-8B 的环境、固定训练/评测入口和资产依赖见 [Qwen3 后训练指南](qwen3-post-training.md)。本页同时保留历史 1.5B 入口，下面标注 T-017～T-029、E-026/E-027、`.venv-gpu` 的命令均属于该历史模型链。

本页不承诺 fresh checkout 一键完成全量复训。历史集群根目录为：

~~~text
/public/home/hpc204166/chenboyu/EnterpriseAgent-RL
~~~

正式运行应保存当前 branch、HEAD、git status、配置 hash 和入口源码快照。历史实验是在未提交的共享工作区完成的，其代码身份以各 run 的源码快照为准，不能仅凭当时的 HEAD 推断。

## 环境约定

| 项目 | 当前约定 |
| --- | --- |
| Python | 源码声明 3.11–3.12；已审核运行环境为 Python 3.12 |
| CPU 依赖 | requirements/cpu-py312-linux.lock |
| CPU 后训练测试 | requirements/cpu-training-py312-linux.lock；增加固定 CPU 版 Torch 2.5.1+cpu 与 NumPy，不安装 CUDA 或下载模型 |
| Qwen3 GPU 依赖 | requirements/qwen3-py312-linux.lock；Torch 2.5.1+cu121、Transformers 4.51.3、tokenizers 0.21.1；.venv-qwen3/bin/python |
| 历史 1.5B GPU 依赖 | requirements/gpu-py312-linux.lock；Torch 2.2.2+cu121、Transformers 4.46.3、tokenizers 0.20.3；.venv-gpu/bin/python |
| 当前 GPU 数值路径 | Qwen3 revision b968826d9c46dd6066d109eabc6255188de91218，FP16 + SDPA math、thinking-off；历史 1.5B revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 |
| 输出原则 | 每次新命令使用一个不存在的显式 run/output 目录；不覆盖旧 artifact |
| 数据原则 | train supervision 只来自获准 train 轨迹；reference SQL/expected answer 不进入模型 messages |

没有安装完整 TRL Trainer；formal GRPO 使用仓库内 local adapter/torch backend。没有 vLLM、FlashAttention-2 默认路径、PG 后端、外部队列或鉴权服务。

GitHub Actions 使用 CPU 后训练锁文件执行 `ruff check .`、`ruff format --check .`、`pytest -q` 和 CPU smoke。单元测试在临时目录生成数据、数据库与虚拟环境；`local_artifacts` 标记仅用于两项历史实验包核对，缺少 Git-ignored 原包时明确 skip，保留普通数据校验、梯度、恢复和服务测试。可通过 `pytest -m local_artifacts` 单独选择历史核对；它不下载或重建原实验。

## 入口与资产总表

表中的“是否产生”描述该入口在实际运行时的可能副作用；T-029 只对标记为 help/静态的命令做了检查。

| 入口 | cwd / Python | 输入来源 | 唯一输出 | GPU | SQL/Verifier | 私有/ignored 资产 | T-029 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| enterprise_agent_rl.cli fixture | repo root；Python 3.12 + CPU lock；.venv-gpu 可用于本机复核 | configs/agent/cpu.toml、源码合成 task | /tmp/earl-cpu-fixture-unique/ 下的 SQLite、trajectory、report/API | 否 | 合成 SQLite；Verifier 使用源码注册表，不访问私有 external DB | 否，输出临时 | 只 help |
| scripts/evaluation/cpu_smoke.py | repo root；CPU env | 源码合成 task、临时目录 | 程序内部 tempfile | 否 | 合成 SQLite/Verifier/API | 否 | 未运行 |
| sft_preflight.py | repo root；.venv-gpu/bin/python | artifacts/sft-data-train-teacher-v2/dataset.jsonl、metadata、model-cache tokenizer | sft-preflight.json | 否 | 不打开业务 DB，不读取 validation/test payload | 是 train artifact/cache；会读 tokenizer | 只 help |
| run_sft.py / sft_train.sbatch | repo root；Slurm + .venv-gpu/bin/python | train-only dataset/metadata、model-cache | 新的 artifacts/sft-pilot-unique/，checkpoint/optimizer/scheduler/scaler/RNG | 是，1 GPU | 训练脚本不执行业务 SQL；输入是 train artifact | 是 dataset/cache；需要 checkpoint 输出 | 只静态核对 |
| current N source | repo root；不重新运行 | T-017-R2 final-step-27/adapter；weight/config SHA 已记录 | 已存在 ignored adapter；不可由 T-029 重建 | 已有证据来自历史 run；本任务 0 | 不重新调用 | 是 N adapter/cache | 只读摘要 |
| run_grpo.py validate-config | repo root；.venv-gpu/bin/python | 显式 GRPO TOML | stdout；handler 不创建 run artifact | 否 | 否 | 只读 TOML | 只静态核对 |
| run_grpo.py cpu-demo | repo root；.venv-gpu/bin/python | grpo-online.toml、4 synthetic tasks | 显式 /tmp/earl-grpo-cpu-demo-unique/ | 否 | 不访问业务 DB/真实 Verifier | 否，fake/tiny | 只静态核对 |
| run_grpo.py train | repo root；Slurm + .venv-gpu/bin/python | train index/evaluation plan/public panel、DB、model cache、parent N | 新的 explicit GRPO run | 是，1 GPU | 真实 rollout SQL/Verifier | 是 | 只静态核对 |
| run_grpo.py cpu-evaluate-frozen | repo root；.venv-gpu/bin/python | frozen plan/panel、validation DB、parent/delta hash inputs | 显式 CPU run | 否 | fake worker，不跑真实 Verifier；仍做 DB/adapter hash preflight | 是；会复制 DB、读 adapter hash | 未运行 |
| run_grpo.py evaluate-frozen | repo root；Slurm + .venv-gpu/bin/python | frozen plan/panel、validation DB、cache、parent N、fixed delta | 新的 explicit frozen run | 是，1 GPU | 真实 SQL/Verifier | 是 | 只静态核对 |
| run_grpo.py score-saved | repo root；.venv-gpu/bin/python | T-024 source run、固定 private task plan、selected validation DB、saved trajectory | 新 closeout 目录 | 否 | 会调用 production Verifier/SQL 一次或幂等复用 | 是 private plan/DB | 未运行 |
| run_sft_n_api.py direct | repo root；Slurm shell + .venv-gpu/bin/python | configs/serving/sft-n.toml、model config/cache、N adapter、train-1 records/DB | 默认 artifacts/service-runs/run-<time>-<nonce>/ | 是，1 GPU | Agent API 会执行 train smoke SQL/Verifier | 是 | 只 help/静态核对 |
| sft_n_api.sbatch | repo root；Slurm submit directory | profile + EARL_MODE/EARL_RUN_DIR | 默认 wrapper 路径 artifacts/service-lifecycle/t027-v1-slurm-<jobid> | 是，1 GPU | 同上 | 是 | 未提交 |
| offline comparison review | repo root；Python 3.12 | 已交付 t025-v1 comparison/metrics/source-index/call-ledger | 不写源包；可写外部摘要 | 否 | 不调用 | 只读完成包摘要 | 只读证据 |

## CPU CLI / fixture

完整 10-task fixture：

~~~bash
cd /public/home/hpc204166/chenboyu/EnterpriseAgent-RL
RUN_DIR=/tmp/earl-cpu-fixture-unique
PYTHONPATH=src .venv-gpu/bin/python -m enterprise_agent_rl.cli --config configs/agent/cpu.toml init-db --database /tmp/earl-cpu-fixture-unique/business.sqlite3
PYTHONPATH=src .venv-gpu/bin/python -m enterprise_agent_rl.cli --config configs/agent/cpu.toml evaluate --database /tmp/earl-cpu-fixture-unique/business.sqlite3 --split all --policy fixture --output /tmp/earl-cpu-fixture-unique/evaluation
~~~

可选的错误恢复 fixture 使用 policy fixture-recovery 和不同 output 目录。不要把 CPU fixture 的任务数量、SQL Action/SQL_OK 或 recovery count 填入 E-026 的 LLM 成功率表。CPU fixture 的公开任务为 train 4、validation 1、test 5；E-026 是 20 个 validation task × 3 arms。

## 历史 1.5B SFT 与 N 的实际来源

### 历史 SFT pilot 入口

源码入口为 scripts/training/run_sft.py，Slurm wrapper 为 scripts/slurm/sft_train.sbatch。wrapper 的默认输入是 dataset.jsonl、dataset-metadata.json 和 artifacts/model-cache。它要求 SLURM_JOB_ID，默认 3 epochs、batch size 1、gradient accumulation 4、learning rate 1e-4、seed 42、LoRA r=8/alpha=16/dropout=0.05，并写新的 artifacts/sft-pilot-<jobid>/。这是一个受控 pilot/训练实现，不是 fresh checkout 自动准备数据、cache、数据库和 N 权重的完整复训器。T-029 没有运行 preflight 或 training。

### 历史 N lineage

E-026/E-027 使用的 N 不是重新由 T-029 生成的 adapter。来源是：

~~~text
artifacts/sft-runtime-aligned-decision-pilot/t017-r2-v1/training/checkpoints/final-step-27/adapter/
~~~

已记录的摘要为 weight SHA-256 e67372df6d805d4ece08e6e698ccbe76d73cc8f83fe96b12f6c7939028f088b9，adapter config SHA-256 bae7f0af3379a71ae24b0386131d1728eea4ed48fef8c7a0f3ba22780ce17c22。T-017-R2 审核确认 27 次有效更新/108 完成微批和 N adapter；后续自然评测/服务使用冻结资产。不要只看到 run_sft.py 存在就声称可以从当前 checkout 无条件重训出同一 N。缺 dataset、metadata、cache、adapter 或其 manifest 时应停止并报告缺项。

## Formal GRPO train 与 CPU demo

只检查配置：

~~~bash
cd /public/home/hpc204166/chenboyu/EnterpriseAgent-RL
PYTHONPATH=src .venv-gpu/bin/python scripts/training/run_grpo.py validate-config --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL --config /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/training/grpo-online.toml --run /tmp/earl-grpo-config-check-unique
~~~

CPU demo：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv-gpu/bin/python scripts/training/run_grpo.py cpu-demo --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL --config /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/training/grpo-online.toml --run /tmp/earl-grpo-cpu-demo-unique
~~~

CPU demo 使用 fake/tiny components，输出的 update、advantage、stage marker、budget 和 comparison 只能证明 CPU 编排契约；不加载 Qwen、tokenizer、业务 DB 或真实 Verifier。

真实 one-GPU train 的调用方法：

~~~bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/public/home/hpc204166/chenboyu/EnterpriseAgent-RL/src /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/.venv-gpu/bin/python /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/scripts/training/run_grpo.py train --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL --config /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/training/grpo-integration-smoke.toml --run /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/artifacts/grpo-formal-entrypoint-validation/t023-v1/unique-run
~~~

该命令读取 train/evaluation manifest 与 DB，加载 fixed revision、parent N，生成 response token/logprob/mask，执行有限两轮更新并写 run-local evidence；T-029 没有提交或运行它。

scripts/slurm/grpo_online.sbatch 的默认 command 是 evaluate-frozen，默认 config 是 grpo-frozen-evaluation.toml；文件名不代表 train。要用 wrapper 调 train，必须显式传 EARL_GRPO_COMMAND=train 和 EARL_GRPO_CONFIG=grpo-integration-smoke.toml，并使用新 run。当前资源和 Job 状态不在本页猜测。

## 冻结评测与离线结果复核

真实冻结评测调用方法：

~~~bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/.venv-gpu/bin/python /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/scripts/training/run_grpo.py evaluate-frozen --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL --config /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/training/grpo-frozen-evaluation.toml --run /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/artifacts/frozen-base-sft-rl-evaluation/unique-run
~~~

它是 eval-only：不构造 train loader/optimizer/scaler，不训练。它仍会读取 validation plan/public panel/DB、cache、parent N 和 fixed delta，并为 Base/N/RL 分别从 fresh Base 加载。Base merge 0、N merge parent 1、RL merge parent 1 + fixed delta 1。

CPU frozen command 只是 fake factory，但仍会读取 plan/panel、复制 DB 和 hash-check adapter：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv-gpu/bin/python scripts/training/run_grpo.py cpu-evaluate-frozen --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL --config /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/training/grpo-frozen-evaluation.toml --run /tmp/earl-grpo-cpu-frozen-unique
~~~

T-029 没有运行上述两条命令。已完成结果复核只读取明确交付包中的 comparison.json、metrics.json、source-index.json、call-ledger.json、report.md 和 manifest.json，不重新调用模型、SQL 或 Verifier。E-026 已审核为 Base 0/20、N 8/20、RL 8/20、unknown 0；N−Base +8/+0.40、RL−N 0、RL−Base +8/+0.40。SQL/worker/Verifier 辅助计数不替代这张业务终点表。

## score-saved 与历史 closeout

score-saved 只适用于固定 T-024/T-025 closeout，不是通用任意 run 评分入口：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/public/home/hpc204166/chenboyu/EnterpriseAgent-RL/src /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/.venv-gpu/bin/python /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/scripts/training/run_grpo.py score-saved --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL --source-run /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/artifacts/frozen-base-sft-rl-evaluation/t024-v1-r2 --private-task-plan /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/artifacts/grpo-online-training/t021-v1/validation/validation-plan.json --arm RL --task-id validation-1-monthly --output /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/artifacts/frozen-base-sft-rl-evaluation-closeout/t025-v1
~~~

该命令会读取私有 task plan、selected validation DB、saved trajectory 和模型证据，并调用 production Verifier/SQL；它不加载模型/tokenizer。T-029 没有运行，避免触发真实/私有 SQL。重复同一 closeout 时由持久 supplement 幂等复用；不能把其一次性固定常量改写成任意 run API。

## N profile 与 launcher

profile 文件为 configs/serving/sft-n.toml，绑定 sft-n-step27、Qwen revision、artifacts/model-cache、T-017-R2 parent adapter、T-021 train-1 benchmark records 和 T-021 train-1 SQLite。

直接 launcher 方法：

~~~bash
cd /public/home/hpc204166/chenboyu/EnterpriseAgent-RL
.venv-gpu/bin/python scripts/serving/run_sft_n_api.py --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL --profile /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/serving/sft-n.toml --mode smoke --auto-ports
~~~

默认 mode=smoke；省略 run-dir 时使用 artifacts/service-runs/run-<time>-<nonce>，显式已有目录拒绝。serve 模式只保持 ready 等待客户端；no-smoke 是 startup-check。GPU 复用方法：

~~~bash
sbatch --partition=gpu4RQ scripts/slurm/sft_n_api.sbatch
~~~

Slurm wrapper 当前默认传入唯一的 artifacts/service-lifecycle/t027-v1-slurm-<jobid> run-dir、mode smoke、soft-cutoff 1320、cleanup-reserve 60 和 auto-ports；它不是由文件名推断出的训练脚本。T-029 只运行 launcher/model_server help，未启动模型/API。

## Fresh checkout 与缺失资产

fresh checkout 包含已提交的源码、公开配置和技术手册，不包含模型、数据库或本地协作过程记录。历史实验的原始日志和源码快照留在本地 ignored artifacts 中。

以下是 ignored 或外部资产，缺失时应显式报告：

- Qwen3 cache、54 条 train 决策、T-032 SFT parent、T-033 token rollout 和 T-034 final delta，以及各自的固定 hash/manifest；
- model cache manifest、Qwen tokenizer/weights；
- 历史 1.5B T-017-R2 N adapter 及 config；
- 历史 1.5B T-023/E-025 fixed RL delta；
- train/evaluation database、benchmark records、public panel、private task plan；
- train-only dataset/metadata、optimizer/scheduler/scaler/RNG checkpoint；
- historical run manifests、trajectory/event/SQL audit and scheduler accounting。

不暗中下载、不从旧 run 重建、不用 CPU fake 替代、不把 manifest 摘要当作已拥有的二进制。业务模型永远不接收 hidden reference SQL/expected answer；private task plan 只能在隔离 Verifier 路径使用。
