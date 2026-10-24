# SFT N 本地模型服务与 Agent API

本手册对应 configs/serving/sft-n.toml、scripts/serving/run_sft_n_api.py、src/enterprise_agent_rl/model_server.py 和当前 service lifecycle。它描述本地 loopback 工程入口，不把 E-027 的两项 train smoke 写成线上部署或泛化评测。

## 服务身份与资产

固定 profile 为 sft-n-step27：

- model：Qwen/Qwen2.5-1.5B-Instruct；
- revision：989aa7980e4cf806f80c7fef2b1adb7bc71aa306；
- dtype：float16；
- attention backend：sdpa_math；
- generation：temperature 0、top_p 1、max_tokens 512、format_retries 1、parse_mode json_fence、seed 17；
- Runtime：seed 17、max_actions 20、max_tool_calls 14、episode timeout 600s、inference timeout 120s、SQL timeout 2s；
- parent adapter：T-017-R2 final-step-27/adapter，已记录 weight SHA-256 为 e67372df6d805d4ece08e6e698ccbe76d73cc8f83fe96b12f6c7939028f088b9，adapter config SHA-256 为 bae7f0af3379a71ae24b0386131d1728eea4ed48fef8c7a0f3ba22780ce17c22。

profile 还绑定 model config、cache manifest、train-1 benchmark records、train-1 database、profile output 和这些输入的 hash。生产身份校验失败应在资产 IO 前停止；不要用 CPU fixture 放宽 identity。

当前固定版本是 Torch 2.2.2+cu121、Transformers 4.46.3、tokenizers 0.20.3。模型服务只实现单卡 Transformers 路径；没有 vLLM、FlashAttention-2 默认路径、完整 TRL Trainer、认证或多租户。

## Help 与静态入口

T-029 实际只执行了 help，未执行 profile preflight、serve、smoke 或 model load：

~~~bash
cd /public/home/hpc204166/chenboyu/EnterpriseAgent-RL
PYTHONPATH=src .venv-gpu/bin/python scripts/serving/run_sft_n_api.py --help
PYTHONPATH=src .venv-gpu/bin/python -m enterprise_agent_rl.model_server --help
~~~

launcher 的 parser 参数是：

~~~text
--repo-root PATH（必需）
--profile PATH（默认 configs/serving/sft-n.toml）
--python/--python-executable PATH
--run-dir PATH
--model-port PORT
--agent-port PORT
--auto-ports
--mode smoke|serve（默认 smoke）
--no-smoke
--ready-timeout 秒（默认 300）
--smoke-timeout 秒（默认 720）
--soft-cutoff 秒（默认 1320）
--cleanup-reserve 秒（默认 60）
--cleanup-grace 秒（默认 15）
~~~

model_server 直接入口的 parser 参数是 profile/repo-root/config/cache/output/port/model-endpoint/health-endpoint/adapter。help 退出发生在 parse_args() 后的 compute-node/profile/model 初始化之前；T-029 的 AST 检查也未发现模块导入级模型/DB调用。

## CPU fixture 与服务合同

CPU fixture 不加载 Qwen。T-028 的正向服务合同使用 profile_id cpu-fixture-sft-n，并把 model TOML、cache manifest/snapshot、adapter bytes/config、两条 train task/private verifier 记录和 SQLite 全部放在 pytest tmp_path。真实 repository artifacts/cache/adapter/database/private plan、非 tmp SQLite 连接和 SQL worker database input 都有拒绝守卫。

T-028 的已有 CPU 审核记录为 25 passed、16.22s、exit 0；这是对 lifecycle/fixture 合同的历史交付证据，本任务没有重跑完整测试 suite。真实 E-027 的 CPU 回归、Ruff、pycompile 和 bash 检查也只引用其原始包，不在 T-029 重新执行。

不要把以下三种路径混写：

| 路径 | 是否读取真实资产 | 是否加载 Qwen | 适用范围 |
| --- | --- | --- | --- |
| CPU fixture tests | 否，tmp-only synthetic assets | 否 | 服务/生命周期/计数契约 |
| direct profile preflight | 是，读取 profile 指向的 cache/adapter/records/DB hash | 否 | 资产完整性准备，不是能力结果 |
| E-027 smoke | 是，cache + N adapter + train-1 DB/records | 是，单 V100 | 两项 train HTTP/API 集成 |

## 计算节点启动方法

从仓库根目录、既有 .venv-gpu 和兼容 Slurm 计算节点启动。方法如下；T-029 没有提交或查询 Job：

~~~bash
sbatch --partition=gpu4RQ scripts/slurm/sft_n_api.sbatch
~~~

scripts/slurm/sft_n_api.sbatch 的源码实际传入：

- profile configs/serving/sft-n.toml；
- mode 为 EARL_MODE，默认 smoke；
- run-dir 为 EARL_RUN_DIR，未设置时使用 artifacts/service-lifecycle/t027-v1-slurm-<jobid>；
- soft-cutoff 默认 1320 秒、cleanup-reserve 默认 60 秒；
- auto-ports；
- .venv-gpu/bin/python；
- Slurm 资源：gpu4RQ、pi_daipeishan、gpuq、1 V100、2 CPU、16G、1500 秒。

文件名只是 wrapper 名称；它通过 scripts/serving/run_sft_n_api.py 启动 model_server 子进程和 Agent API 子进程。若改分区，必须重新核对实际 GPU 型号/数量、CPU/内存、时限、账户/QoS 和预算兼容性。

已有计算节点 shell 的直接方法：

~~~bash
.venv-gpu/bin/python scripts/serving/run_sft_n_api.py \
  --repo-root /public/home/hpc204166/chenboyu/EnterpriseAgent-RL \
  --profile /public/home/hpc204166/chenboyu/EnterpriseAgent-RL/configs/serving/sft-n.toml \
  --mode smoke --auto-ports
~~~

直接 launcher 不传 run-dir 时，在 artifacts/service-runs/ 下用原子目录创建 run-<time>-<nonce>；连续调用不扫描、复用或覆盖旧 run。显式已有 run-dir 被拒绝。--python-executable 保留传入 venv symlink 的 spelling，实际 argv 写入 commands.json。

## smoke、serve 和 startup-check

- smoke（默认）：启动 model + Agent API，等待两个 health，执行固定的 train-1-schema/train-1-empty smoke，写 panels、trajectory、summary/checks 后收尾。
- serve：只保持 ready 并等待客户端请求；不自动请求 benchmarks，不自动提交任务；直到信号、子服务早退或 run-global deadline。
- no-smoke：只能和默认 mode=smoke 一起使用，语义为 startup-check；ready 后立即退出，不是持续 serve。

生命周期使用一个 supervisor monotonic clock，覆盖 profile/preflight、目录/进程 setup、ready、HTTP 提交/轮询、serve 等待和 cleanup。默认 soft cutoff 1320s，cleanup reserve 60s；ready/smoke timeout 还受全局剩余预算约束。TERM/INT 只作用于本 launcher 创建的进程组，执行 TERM → bounded wait → KILL → wait。

| 情况 | 持久状态 | 退出码 |
| --- | --- | ---: |
| smoke 完成 | COMPLETED | 0 |
| startup-check ready 完成 | STARTUP_CHECK_COMPLETED | 0 |
| ready/轮询/serve 达到 deadline | DEADLINE_EXCEEDED | 1 |
| 子服务早退/其他 supervisor 错误 | CHILD_PROCESS_FAILURE 或 FAILED | 1 |
| TERM | STOPPED_BY_SIGNAL | 143 |
| INT | STOPPED_BY_SIGNAL | 130 |

每个终态任务先写 panels/<task-id>.json；第二项失败、deadline 或信号不会删除第一项，未完成项为 unknown/technical_unknown。

## 客户端与私有评分边界

服务 ready 后客户端只发送公开 prompt 和公开 benchmark_task_id：

~~~bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/models
curl -s http://127.0.0.1:8000/benchmarks
curl -s -X POST http://127.0.0.1:8000/tasks \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: example-schema' \
  -d '{"prompt":"先使用 schema 确认客户表字段，列出客户名称和区域编号，返回 customer, region_id，按客户名排序。","benchmark_task_id":"train-1-schema"}'
curl -s http://127.0.0.1:8000/tasks/<job_id>
curl -s http://127.0.0.1:8000/tasks/<job_id>/trajectory
~~~

公开 /benchmarks 只返回 task_id、prompt、split；reference SQL、隐藏约束和 expected answer 只给独立 Verifier，不能复制进 model messages。/models、/health、job 和 trajectory 记录 sft-n-step27、Qwen revision 和 adapter 摘要。technical_failure/technical_unknown 不是业务 false；SQL 错误、invalid Action、预算终止是已持久业务结果。

## Run 产物、计数和证据

direct launcher 的 run 目录至少包含：

- commands.json：实际 argv、解释器、模式和退出信息；
- process/records.json：子进程和回收记录；
- model/generations.jsonl：模型 generation event；
- api/trajectories/：最终 trajectory；
- api/sql-audit.jsonl：SQLTool.execute audit；
- panels/：公开 task/result 摘要；
- verifier-private/：参考 SQL/约束等私有字段；
- summary.json、metrics.json、manifest.json、allocation-ledger.json：结果、hash 和 wall/allocation 分离。

T-028 关闭后的计数规则是：model event 优先于 trajectory model_attempts；SQL audit 优先于 trajectory SQL step；fallback 互斥，不能相加；Verifier 按最终 trajectory 中 non-null reward 计数；缺来源为 null 并给 reason。gpu_calls 没有独立来源时为 null，不能把 wall clock 或没有 scheduler 记录写成 0。

## 已验证结果与限制

E-027/T-026 的真实 run 为固定 N、单 V100、Job 20419673、gpu2Q/gpu201、192 秒、26 个 model request/event、15 次 Runtime tool attempt、2 次 Verifier 调用、technical unknown 0。train-1-schema 为业务 false/action_budget_exhausted，train-1-empty 为业务 true/completed。它证明两项 train 的本地 loopback HTTP 链路，不改变 E-026 的 20 项冻结结果。

E-026/T-025 的独立冻结结果是 Base 0/20、N 8/20、RL 8/20；RL 使用 parent N + fixed final delta。T-028 的 25 passed 是 CPU lifecycle/fixture 证据，不能写成 V100 长期服务或生产多租户证据。

当前没有认证、外部队列、PG、跨 Job 自动累计资源守卫或公网部署。服务结束后不会自动启动训练、完整 validation/test、多 seed 或常驻部署。
