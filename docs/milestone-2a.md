# Milestone 2A 工程交付与运行手册

本阶段在 CPU MVP 之上新增真实 Transformers 服务、模型输出审计和冻结业务基准。**真实 GPU / Qwen / Base 指标均 Pending**；收费作业尚无用户授权。CPU 协议测试中的 HTTP/engine 测试替身不属于模型结果。

## 准备与依赖

```bash
bash scripts/setup/bootstrap_gpu.sh
# 缓存下载不加载权重；先确认 quota 和网络。
.venv-gpu/bin/python scripts/setup/cache_qwen.py --cache "$PWD/artifacts/model-cache"
# 在一个新的目录重建数据库，并与已冻结清单逐字核对。
.venv/bin/python scripts/evaluation/prepare_m2a.py
```

GPU 依赖环境不使用 CPU venv 的 site-packages，不修改原 CPU lock。GPU lock 固定所有传递依赖，`.venv-gpu`、权重、缓存、SQLite 与运行轨迹均忽略提交。模型下载仅 safetensors/配置/分词文件，固定 revision；缓存清单记录逐文件 SHA-256。模型服务 `local_files_only=True`，缓存缺失会失败，不在收费计算时间内隐式联网下载。

服务必须位于 Slurm 分配的计算节点：检查 job ID、nodelist 中的实际 hostname、GPU 分配及单一可见 CUDA 设备；在检查前不 import Torch 或加载模型。HTTP 绑定 127.0.0.1，单 worker、串行生成，繁忙时结构化返回 model_busy。

## 收费作业：仅获明确授权后执行

以下是后续命令示例，**本轮未执行**；账户/QOS 须按授权指定。每次作业单卡 gpu2Q，不使用八卡。没有编写会自动提交后续作业的脚本。

```bash
sbatch --account=<获准账户> --qos=<获准QOS> scripts/slurm/gpu_inventory.sbatch
sbatch --account=<获准账户> --qos=<获准QOS> scripts/slurm/qwen_single.sbatch
sbatch --account=<获准账户> --qos=<获准QOS> scripts/slurm/qwen_agent.sbatch
sbatch --account=<获准账户> --qos=<获准QOS> scripts/slurm/qwen_base.sbatch
# Base 默认 validation。冻结 prompt/生成/预算配置后才开展最终测试：
EARL_EVAL_SPLIT=test sbatch --account=<获准账户> --qos=<获准QOS> scripts/slurm/qwen_base.sbatch
```

首次运行使用默认数据库目录；重复重建须用 `--database-root` 指定新目录，并在作业中用 `EARL_DATABASE_ROOT` 指向该目录，已有库不会覆盖。

`EARL_MODEL_CACHE` 和 `EARL_GPU_PYTHON` 可覆盖缓存和解释器；`EARL_AUDIT_IMAGE` 指向已有 Singularity 镜像，以实测 `--nv`。未提供镜像则明确记录 pending。模型实验每个 job 使用新输出目录，拒绝覆盖已有证据。

单轮最多 64 新 token，验证实际内容生成。Agent smoke 使用现有 `top-2025`，要求先确认订单数、继续查 Top-3、check、finish，并由原独立 verifier 打分；失败时作业返回非零且保留实际轨迹。Base job 评测选中 split 的 20 条任务；模型业务失败是有效的评测结果，独立 verifier 出错则报告 incomplete，不能作为完整基线。

Base 作业上限 1 小时，单任务最坏 600 秒；20 个任务全部接近超时可能无法在时限内完成。中断时保留已经生成的轨迹、部分报告和服务日志，`complete=false`；应根据获准 smoke 的实际延迟决定后续作业时限，不把部分成功率当最终测试指标。

## Runtime 与输出证据

`configs/agent/qwen.toml` 将任务整体预算设为 600 秒、单次 decide（包含格式重试）设为 120 秒，SQL 仍为独立 2 秒；CPU 默认整体 30 秒保持。模型请求 timeout 由推理预算和 episode 剩余时间共同限定，已移除客户端固定 30 秒上限。

`GenerationSettings` 配置 temperature、top_p、max_tokens、seed、parse_mode 和 format_retries。默认贪心生成，每次 decide 最多一次格式重试。解析只接受一个完整 JSON Action；可选去除包围整个输出的 JSON fence，不从说明文字中提取片段、不补字段、不修 SQL、不替模型选择工具。未知 kind、参数类型/键错误、空输出和非法 JSON 分别归类。业务上的无效证据或 SQL 错误由原环境反馈，后续决策仍由模型生成。

轨迹 v1.1 保存全部 `model_attempts`：原始文本、解析 Action、格式有效性、重试编号、token 计量、耗时、服务端 telemetry 与错误；失败且没有合法 step 的输出也保留。每个执行 step 另关联对应尝试；保存时校验实际 Action 与原始文本解析结果相同。v1.0 CPU 轨迹仍可读取和重放。参考 SQL、oracle 和 split 标签不进入模型消息；finish 的 rows 继续由真实工具证据提供。

模型服务在返回 HTTP 前持久记录生成请求和输出。加载错误、OOM、inference_timeout、empty_output、invalid_json、invalid_action、invalid_action_arguments、context_length_exceeded 及传输异常均有明确记录。生成 deadline 在 token 间检查，HTTP 另设墙钟超时；卡住的 CUDA kernel 由作业时限/进程终止兜底，不声称 token deadline 能抢占任意 kernel。supervisor 在正常结束、失败、SIGTERM/INT 时先 TERM 自己创建的模型进程组，超时后 KILL 并 wait；Slurm 最终回收作业。测试使用真实短生命周期子进程验证回收，不加载模型。

产物位于 `artifacts/qwen-<mode>-<job>/`：

| 文件 | 内容 |
|---|---|
| execution.json | commit/dirty、Slurm/hostname、完整配置、状态、服务退出码与回收结果 |
| pip-freeze.txt / scheduler-job.txt / nvidia-smi.txt | 依赖、资源分配及驱动/设备原始记录 |
| server/model-load.json | 模型 revision、实际 dtype/attention、load latency、加载显存峰值 |
| server/generations.jsonl | 每次请求的真实原始输出/usage/速度/峰值显存或错误，包括部分超时输出 |
| server/startup-errors.jsonl / server.log | 启动与兼容性故障 |
| smoke/ 或 base/trajectories/*.json | 模型 Action、完整工具反馈、最终 Reward 和每任务失败轨迹 |
| base/report.json | 聚合指标、逐任务 reward、冻结清单哈希、完成状态与验证器异常 |

## 冻结业务基准

`benchmarks/m2a/manifest.json` 共 60 条任务，6 个数据库实例，train/validation/test 各 20 条、2 个实例。每条包含来源、schema/data 版本、reference_sql、split、模板、schema 与数据行的规范化 SHA-256、独立结果指纹与任务指纹。manifest SHA-256：`232ed8b69cdf8ba9a63d40413ae96794e00b22f339bb3f5041acf3dd71a67a0d`，由同目录 `.sha256` 固定。

冻结清单使用逻辑 schema/数据指纹，避免 SQLite build/header 差异使相同业务数据失配；生成目录的 materialization.json 另记录 SQLite 版本与每个数据库文件的字节哈希。轨迹仍使用文件字节哈希进行严格 replay。

各 split 使用不同 seed、年份（2021/2023/2025）、客户和商品名称；8 类公共模板（schema、区域 JOIN、Top-3 多步、指定客户、指定商品、空结果、复杂约束、错误恢复）及每个 split 2 类独占模板。测试独占去重客户数和订单件数约束。原 CPU 的 10 个任务保留作功能 smoke，不与冻结测试集混作泛化评估。

所有 60 个 reference_sql 均实际通过原受限 SQLTool 执行，并与只读取基础行、在 Python 中独立聚合的 oracle 逐行核对；没有从参考 SQL 的输出反向抄造标签。错误恢复任务公开要求先执行历史字段查询并依据实际错误和 schema 修正；verifier 要求出现真实 SQL 失败后恢复。schema 任务必须调用 schema；多步 Top-3 必须出现独立核对的 paid_orders probe。

这些任务仍是小型合成、共享五表 schema 和部分模板的基准，不能推断生产库或完全跨域泛化。最终 test 已在训练前冻结；不得把其 reference_sql、轨迹或失败案例用于 SFT/GRPO 或提示词调参。后续仅用 train 轨迹构建 SFT，用 validation 调参。修改测试定义必须发布新 benchmark 版本，不能静默更新此清单。

## 指标口径

保留原 Task Success Rate、SQL executable rate、Tool Error Rate、Average Tool Calls、Average Episode Latency 和 token/配置成本统计。新增：

- `action_format_validity_rate`：所有实际返回文本（含重试和非法文本）中通过配置解析器的比例；`strict_action_format_validity_rate` 不计 fence 规范化成功。传输失败无文本，不进入文本格式分母，单列 request/output 数。
- `generation_throughput_tokens_per_s`：有服务端测量的输出 token 总数 / 对应 generation latency 总数，含 prefill/生成及同步，不宣称纯 decode 速度。
- `gpu_peak_allocated_bytes` / `gpu_peak_reserved_bytes`：逐生成峰值的最大值；加载峰值另见 model-load.json。它们是 Torch allocator 指标，不等于 nvidia-smi 整卡占用。
- timeout episode 数、inference timeout 次数、OOM 次数、invalid output 次数；重试仍计入无效输出总数，不能用最终解析成功掩盖首次失败。

终局 token 计量缺失时保留 unknown/null，不伪造 0。fixture 报告明确 `policy_kind=deterministic_fixture` / `llm_capability_result=false`；真实本地模型报告为 local_llm。禁止混合两种报告。验证器错误时报告完整性为 false，metrics 只对应已成功评分的 episodes，并公开 scheduled/scored 分母。

## 后续 SFT / GRPO

先获准完成设备审计、真实单轮和多轮 smoke，再冻结 Base 配置并跑验证/测试报告。只导出 `split=train`、独立 verifier 成功且原始生成可追踪的轨迹；训练前检查 task/database/manifest 指纹、去重及 train-only 约束。需要补充 chat template 的 assistant-only token mask，再讨论 LoRA SFT。GRPO 还需要真实 token IDs/logprobs、response mask、分组 rollout 和可恢复状态；本轮没有实现或运行正式训练，也没有安装完整 verl/Agent-STAR。
