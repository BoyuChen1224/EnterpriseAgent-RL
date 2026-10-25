# GRPO 训练与冻结评测

本手册先介绍当前 Qwen3-8B 固定批次更新，再保留历史 Qwen2.5-1.5B 在线入口。两条路径共享本地 GRPO loss/backend，但使用各自的模型、配置、数据和父 Adapter；命令名称不代表它们可以互换。

## 当前 Qwen3-8B 固定批次 pilot

模型为 `Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`，FP16、SDPA math、thinking-off。环境和资产依赖见 [Qwen3 后训练指南](qwen3-post-training.md)。

| 阶段 | 入口 | 配置 |
| --- | --- | --- |
| 冻结 SFT 分组采样 | `scripts/evaluation/run_qwen3_grpo_rollout.py` | `configs/evaluation/qwen3-grpo-rollout.toml` |
| 批次校验、更新、重载、SFT/RL 对照 | `scripts/training/run_qwen3_grpo_pilot.py` | `configs/training/qwen3-grpo-update-pilot.toml` |
| 单卡 Slurm 分阶段 wrapper | `scripts/slurm/qwen3_grpo_update_pilot.sbatch` | `T034_STAGE`、`EARL_T034_RUN`、`EARL_T034_CONFIG` |

入口检查只需执行：

```bash
PYTHONPATH=src .venv-qwen3/bin/python scripts/evaluation/run_qwen3_grpo_rollout.py --help
PYTHONPATH=src .venv-qwen3/bin/python scripts/training/run_qwen3_grpo_pilot.py --help
```

更新入口包含 `prepare`、`tiny`、`train`、`reload`、`compare`、`record-allocation` 和 `finalize`。其中 `train`、`reload`、`compare` 是独立 GPU 阶段；该入口没有新的 rollout 阶段。固定来源由配置中的文件路径和 SHA-256 校验，不能仅更换输出目录后假定资产齐全。

### 采样与优化目标

- 4 个 train task，每组 4 个真实多轮 episode，共 16/16 正式评分、unknown=0。2/4 个组存在奖励差异，另外两组全成功、优势为零。
- 保存 118 条物理请求的原生 prompt/completion IDs、old policy logprob、采样 logprob 和 response mask；70 条有非零优势、48 条为零。
- 组优势使用 sample standard deviation + `1e-4`；同一 episode 的请求继承其组优势。仅 response token 进入 policy loss，工具 observation 保持在上下文中。
- 使用 clipped policy ratio，epsilon=0.2、beta=0；本轮没有 KL 惩罚。AdamW learning rate 为 `2e-6`，新 LoRA 为 rank 8/alpha 16/dropout 0，Base 和 parent 冻结。
- 两次更新复用同一冻结 batch 和 old logprobs。每次对 70 条非零优势记录累计 backward；完整归一化分母为 4745 token，包含零优势记录。实际完成 140 次 backward、2 次成功 optimizer.step，AMP skip=0。

### 加载与结果

独立重载按 **fresh Base → parent SFT merge → GRPO delta merge**，两次 merge 各执行一次。父 SFT SHA 为 `413d1b0211ba93b2df322dcdaeb1fa1c3f019210c8ded510a2920e5ad6e114e9`，final delta SHA 为 `dc46efecbd519142ccca44fa3291861177f299bf0e599cced65094fea9e1260a`。禁止把 delta 当作脱离父 SFT 的独立模型。

fresh SFT 与 RL 均在 20 项冻结 validation 上完成正式评分，均为 13/20，unknown=0；13 个共同成功、7 个共同失败，gain/loss 均为 0。更新、重载和完整评测成立，本轮没有额外业务收益。它是固定批次两更新的有限试验，不代表多轮在线训练、长训效果或跨库泛化。

## 历史 1.5B 在线入口与结果

以下各节记录 `scripts/training/run_grpo.py`、1.5B TOML、E-026 和固定 closeout；其中的 N、路径、默认 Slurm 参数与成功率均属于历史 1.5B 链，不替代上面的 Qwen3 配置。

formal GRPO 已有有限真实 rollout、TokenRecord、响应 token mask、组内优势、更新/恢复和 T-023/E-025 final-delta 证据；E-026/T-025 已完成同一 20 项 validation panel 的 Base/N/RL 冻结对照。当前结果：

| arm | planned / scored / unknown | success |
| --- | ---: | ---: |
| Base | 20 / 20 / 0 | 0/20 |
| N | 20 / 20 / 0 | 8/20 |
| RL（parent N + fixed delta） | 20 / 20 / 0 | 8/20 |

N−Base 为 +8/+0.40（+40 个百分点），RL−N 为 0。该结果只适用于单 seed、共享 schema/template 的冻结功能面板，不证明跨库/模板泛化，也不证明 GRPO 普遍无效。

实现是仓库内 local GRPO adapter + torch backend，不是安装完整 TRL Trainer、vLLM、PEFT/Accelerate、分布式队列或长训服务。CPU 检查不替代真实 GPU 结果。

## CLI 子命令与副作用

所有 formal 子命令都要求显式 repo-root、config 和 run，除 score-saved/finalize/replay 的专用参数外不猜测旧目录。

| 子命令 | 实际行为 | 资产/副作用 |
| --- | --- | --- |
| validate-config | 只解析并校验 TOML，打印 model/budget，标记 asset verification deferred | 不加载模型、不访问 DB、不调用 Slurm；只读配置 |
| cpu-demo | synthetic tasks + tiny Torch policy + FakeRolloutEngine/FakeWorker，走在线训练编排和 CPU fake evaluation | 写显式 run；不加载 Qwen/tokenizer、不访问业务 DB/真实 Verifier |
| train | 真实 one-GPU online rollout → update → optional N/RL evaluation | 读取 train/evaluation index、DB、cache、parent adapter；需要兼容 GPU/Slurm；写 run-local evidence |
| cpu-evaluate-frozen | fake 三臂严格评测 contract | 不加载模型/tokenizer/真实 Verifier，但仍读取 frozen plan/public panel、复制 DB，并做 parent/delta hash preflight；没有这些 ignored assets 会失败 |
| evaluate-frozen | eval-only fresh Base/N/RL，跳过 train loader/optimizer/training loop | 读取 20 项 plan、panel、validation DB、cache、parent N 和 fixed delta；需要 GPU/模型/业务 Verifier |
| finalize-frozen | 对中断 frozen run 补齐缺失结果为 unknown，离线聚合和写 evidence | 不执行模型或 Verifier；只操作已有 run |
| finalize | 对一个明确 verifier index 聚合 raw_saved/scored/unknown | 不 glob 轨迹；写指定 output |
| replay-t021 | 只读交付的 T-021 JSON/JSONL formal index | 不导入历史 Python、不访问 DB/权重、不重新调用模型 |
| score-saved | 对一个明确保存的 trajectory 做 CPU production Verifier closeout | 读取固定 source run/private task plan/selected DB，调用 Verifier/SQL；不加载模型/tokenizer/训练 |

这张表以 src/enterprise_agent_rl/grpo/cli.py 的 parser 和 handler 为准；不要仅根据 scripts/slurm/grpo_online.sbatch 的文件名推断它会训练。

## 配置和模型身份

当前 online config 的固定约束包括：

- Qwen/Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306；
- FP16 base、sdpa_math attention、LoRA rank 8/alpha 16/dropout 0 和固定 target modules；
- 两轮、每轮两次 update、temperature 1/top_p 1/top_k 0/max_tokens 512 的 rollout；
- evaluation generation 为 greedy temperature 0/seed 17；
- 显式 job/run budget、call limits、train/evaluation index、public panel 和数据库路径。

正式 frozen config 是 configs/training/grpo-frozen-evaluation.toml。它绑定：

- parent N adapter 及 weight/config hash；
- T-023/E-025 的 RL final-delta 及 hash；
- 20 个 validation task、两个 validation DB 的 physical/logical fingerprint；
- tokenizer template 摘要、Verifier/Action/worker limits 和 one-GPU Slurm 参数。

RL 必须按 parent N + fixed final-delta 加载；不能从 Base 另找一个“独立 RL 模型”替代。没有 parent/delta/cache/DB/plan 时应明确报缺资产，不下载、不重建、不把 CPU fake 当真实结果。

## CPU fake/tiny 入口

先在仓库根目录检查 parser：

~~~bash
cd /public/home/hpc204166/chenboyu/EnterpriseAgent-RL
PYTHONPATH=src .venv-gpu/bin/python scripts/training/run_grpo.py --help
~~~

CPU online demo：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src .venv-gpu/bin/python scripts/training/run_grpo.py cpu-demo \
  --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL \
  --config /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/training/grpo-online.toml \
  --run /tmp/earl-grpo-cpu-demo-unique
~~~

它用 4 个 synthetic tasks、tiny Torch 参数、FakeRolloutEngine/FakeWorker，仍检查 rollout、group advantage、optimizer/scheduler state、stage marker 和 CPU comparison。输出只能证明 CPU 编排契约。

CPU frozen contract：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src .venv-gpu/bin/python scripts/training/run_grpo.py cpu-evaluate-frozen \
  --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL \
  --config /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/training/grpo-frozen-evaluation.toml \
  --run /tmp/earl-grpo-cpu-frozen-unique
~~~

这个命令不是零资产检查：handler 会读取 evaluation plan/public panel，复制 validation DB，并按配置 hash-check parent N 与 fixed delta；只有模型/真实 verifier 执行被 fake factory 替换。T-029 未运行该命令，也未运行任何 preflight。

validate-config 适合只检查 TOML：

~~~bash
PYTHONPATH=src .venv-gpu/bin/python scripts/training/run_grpo.py validate-config \
  --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL \
  --config /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/training/grpo-online.toml \
  --run /tmp/earl-grpo-config-check-unique
~~~

它不访问 model/database，真实 asset verification 延后到 GPU load；不要把 PASS 写成模型或数据资产已取得。

## 真实 online train 入口

真实 one-GPU 训练方法如下；T-029 只记录命令，没有执行：

~~~bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/public/home/hpc204166/chenboyu/EnterpriseAgent-RL \
.venv-gpu/bin/python scripts/training/run_grpo.py train \
  --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL \
  --config /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/training/grpo-integration-smoke.toml \
  --run /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/artifacts/grpo-formal-entrypoint-validation/t023-v1/unique-run
~~~

上面的 config 是 T-023/E-025 两组 integration smoke，不能替代 E-026 的 20 项冻结面板。正式 train 会读取 train index/evaluation plan/public panel、复制 DB、加载 Base+parent N、采集 rollout token/logprob/mask、执行有限更新并写 N/RL 评估。它需要 Slurm one-GPU 资源；不在登录节点运行。

### Slurm 默认语义

scripts/slurm/grpo_online.sbatch 的当前源码默认值是：

- config：configs/training/grpo-frozen-evaluation.toml；
- command：evaluate-frozen；
- Python：.venv-gpu/bin/python；
- partition：gpu4RQ，account pi_daipeishan，qos gpuq，1 GPU、2 CPU、16G、50 分钟；
- run：artifacts/frozen-base-sft-rl-evaluation/t024-v1，若已存在则追加 r2/r3。

所以它默认是冻结评测，不是 online train。若确实要调用 train，必须显式覆盖 config 和 command，并保留唯一 run：

~~~bash
EARL_GRPO_COMMAND=train \
EARL_GRPO_CONFIG=/public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/training/grpo-integration-smoke.toml \
EARL_GRPO_RUN=/public/home/hpc204166/chenboyu/EnterpriseAgent-RL/artifacts/grpo-formal-entrypoint-validation/t023-v1/unique-run \
sbatch --partition=gpu4RQ scripts/slurm/grpo_online.sbatch
~~~

该命令是未来兼容资源下的调用方法；本任务 GPU、Slurm submission/query 和 Job 均为 0。pending 时只可在满足 GPU/CPU/内存/QoS/预算的兼容分区间迁移，旧 run 和累计 budget 不重置。

## Frozen Base/N/RL 评测

eval-only 方法：

~~~bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv-gpu/bin/python scripts/training/run_grpo.py evaluate-frozen \
  --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL \
  --config /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/training/grpo-frozen-evaluation.toml \
  --run /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/artifacts/frozen-base-sft-rl-evaluation/unique-run
~~~

该路径不构造 train loader、optimizer、scaler、LoRA training loop；每个 arm 都从同一 fresh Base 开始，Base merge 0 次、N merge parent 1 次、RL merge parent 1 次再 merge fixed delta 1 次。所有 arm 使用同一 Runtime/Verifier boundary、greedy temperature 0/seed 17 和 20 项 task set。

### 两臂独立 gate 与三臂完整状态

聚合器输出 arm contract、pair status 和 global status：

- N−Base 只看 Base 与 N；
- RL−N 只看 N 与 RL；
- RL−Base 只看 RL 与 Base；
- global three-arm COMPLETE 才要求三臂和三个 pair 均完整。

每个 arm 的完整 contract 要求 declared/planned/scored/qualified=20、unknown=0、task IDs 与 protocol fingerprint 匹配。业务失败是 scored=false，技术证据缺失才是 unknown。缺失 RL 不能遮蔽完整 N−Base；global 仍可 incomplete。

每次模型调用先持久化 INTENT，再绑定 response/event/TokenRecord/Action/trajectory。response mask 的 prompt token 为 0、response token 为 1；raw policy 与 behavior sampling log-probabilities 逐 token 保存。组内 reward 用 sample standard deviation + 1e-4 计算 advantage；零方差组是合法零 advantage 并跳过更新。SQL Action、Runtime worker、Verifier SQL、业务 success 和 technical unknown 分开。

## score-saved：只用于固定 closeout

score-saved 是 CPU-only production Verifier scoring，不是任意 run 的通用评分服务。当前实现把 T-024/T-025 的 source manifest、trajectory、task ID、database fingerprint、policy identity 和固定 arm/task 常量绑定在 saved_scoring.py；当前审核过的目标是 RL/validation-1-monthly。

历史 closeout 复核方法：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src .venv-gpu/bin/python scripts/training/run_grpo.py score-saved \
  --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL \
  --source-run /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/artifacts/frozen-base-sft-rl-evaluation/t024-v1-r2 \
  --private-task-plan /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/artifacts/grpo-online-training/t021-v1/validation/validation-plan.json \
  --arm RL \
  --task-id validation-1-monthly \
  --output /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/artifacts/frozen-base-sft-rl-evaluation-closeout/t025-v1
~~~

它只对已保存 trajectory 做独立 Verifier/SQL closeout，输出独立 call-ledger 和 merged view，不改变 source run；重复同一 closeout 会复用已持久 supplement。不要把这条命令改写成“任意 run 可评分”，也不要把它与 evaluate-frozen 或 train 混为一条路径。

## 持久证据、恢复和离线命令

每个 round 写 rollout fingerprint、raw episode/token/score records、dataset index、policy snapshot、training state 和 metrics。budget-state.json 记录 allocated GPU seconds、phase/episode/update 计数和恢复状态；queue time 不冒充 GPU 使用。

- finalize-frozen 只对已有 interrupted run 补 unknown 并离线聚合，不执行模型/Verifier。
- finalize 只接受一个显式 verifier index；raw_saved 不是 scored，冲突重复会失败，缺轨迹为 unknown。
- replay-t021 只读取已经交付的 JSON/JSONL formal evidence，不运行历史代码。
- unknown 不自动转 false；SQL 可执行不自动转业务成功；loss/update 完成不自动转能力提升。

冻结三臂比较的本地原始证据位于 artifacts/frozen-base-sft-rl-evaluation-closeout/t025-v1/；原始实验包不随源码提交，业务结果汇总见 README。
