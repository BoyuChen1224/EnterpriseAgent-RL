# CPU 验收记录

日期：2026-10-15（Asia/Shanghai）。功能代码提交：`42bc006867c8654461b3feaf93b38e44edea7359`。本报告区分最终通过结果、开发期间失败和仍未运行的项目。

## 最终 CPU 自动测试

实际提交：`sbatch scripts/slurm/cpu_smoke.sbatch`。

- Job ID：**20383967**。
- 分区/账户：freecpuQ / free，normal QOS。
- 申请：1 节点、1 task、1 CPU、2 GB、上限 5 分钟，无 GPU。
- 实际节点：**node0857**；Slurm 最终状态 **COMPLETED**，ExitCode **0:0**，Elapsed **00:00:41**。
- 作业内 `.venv/bin/python -m pytest -q`：**48 passed in 12.00s**。
- 作业内 `.venv/bin/python scripts/evaluation/cpu_smoke.py`：**passed**；5 个 test 任务全部成功、5 条错误恢复轨迹全部重放匹配、API 提交/状态/轨迹/奖励闭环成功。

原始 Slurm 日志保留在本地 `slurm-earl-cpu-smoke-20383967.out`，未提交 Git。早期免费验收 20383950 在 node0933 为 43 passed；随后 20383962 在 node0933 为 46 passed。最终以 20383967 的代码与新增边界测试为准。

测试覆盖的实际场景：

- 固定 seed 可重复生成五表数据、外键一致性、已有文件不覆盖、任务 split 不重叠。
- Schema/外键与字典发现；SQL 合法聚合、独立 Python 业务核算对照。
- 19 类写入/系统表/文件函数/ATTACH/扩展/多语句 SQL 拦截，数据库字节指纹不变。
- 查询行数/字节/列名限制、真实高开销只读 SQL 超时，以及 Schema 超时异常路径。
- 所有 10 个任务的多轮 plan→schema→dictionary→sql→check→finish。
- SQL 字段错误恢复、重复调用诊断、失败不拿正确性奖励。
- 中间业务计数必须与独立 verifier 相符，重复最终查询不能刷次数满足约束。
- 工具/action 预算、episode timeout、错误证据/截断证据拒绝、reset 状态清理。
- 轨迹保存/读取/JSONL、序号/版本/证据篡改拒绝、数据库哈希检查及 action replay。
- 同条件评测比较、fixture 标签、不同条件拒绝、未知真实模型 token/费用为 null。
- API 任务、版本、比较、状态、轨迹、错误请求、重启状态、批量容量失败后释放。
- 本地模型 HTTP 消息/token 协议与登录节点/外部端点限制；使用模拟 HTTP 响应，未加载模型。

## CLI、API、静态检查与依赖

已运行并通过：

```bash
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check
.venv/bin/ruff check .
.venv/bin/ruff format --check .
git diff --check
.venv/bin/earl --config configs/agent/cpu.toml init-db
.venv/bin/earl --config configs/agent/cpu.toml evaluate --split all --output artifacts/all-tasks
.venv/bin/earl compare <fixture-report.json> <recovery-report.json>
```

`--split all` 功能评测曾对 10 个任务获得 10/10，不是训练或泛化实验。最终 test split 对照由作业 **20383968** 在 **node0857** 生成，COMPLETED / ExitCode 0:0 / 5 秒，正常与恢复策略均为 5/5，已保存到 `experiments/cpu-mvp/`。

还在独立子进程实际启动 `.venv/bin/earl serve`，使用真实 loopback Uvicorn HTTP 服务运行 `.venv/bin/python examples/api_client.py`，成功提交 Top-3 任务、轮询 completed、获取成功 reward 与完整轨迹，随后终止服务。输出保留于本地 `artifacts/api-client-output.txt`，没有遗留演示服务器。

Ruff 最终报告 All checks passed、18 files already formatted；pip check 报告 No broken requirements found。CI 工作流已加入 `.github/workflows/cpu.yml`，但本地完成不等于已观察到 GitHub Actions 运行成功。

## 开发期间失败与实际修复

初轮 CPU smoke 发现只读 URI 缺少 `uri=True`；后续测试发现 SQLite COUNT/覆盖索引的特殊 READ 授权参数。均已修复并增加/保留回归测试。

登录节点一次 `.venv/bin/pytest -q --cov=enterprise_agent_rl --cov-report=term-missing` 得到 **45 passed、2 failed**：普通参考查询超时导致独立核算和评测测试失败。同阶段恢复 CLI 对照曾为 4/5；失败轨迹显示同步 Schema 读取 46.6 秒后才返回，触发 episode_timeout。该失败保留于 `artifacts/release/`，没有当作最终通过数据提交。

修复将 Schema 发现也移入有墙钟超时的受限子进程，并增加故障测试；最终在 freecpuQ 计算节点验收通过。没有通过关闭限制、删除失败用例或把失败任务换成成功任务来获得最终结果。共享文件系统和调度负载仍可能造成真实 timeout，应保留并报告，不保证每台机器固定 latency。

覆盖率工具对 stdlib SQL 子进程和手动 CLI 路径未完整插桩，因此不以那次覆盖率百分比作为验收标准；SQL 与 Schema 的边界由真实子进程集成测试验证。

## 未运行

没有提交 GPU 作业；没有确认设备级显存/驱动/CC；没有加载 Qwen、运行 FP16 推理、SFT、GRPO、checkpoint 恢复、GPU 性能比较、BIRD 评测或公开部署。CPU fixture 的 token 为 0，配置单价为 0，不代表真实 LLM 或集群费用为零。Base/SFT/RL 三组结果尚不存在。

## Milestone 2A CPU 工程验证

本节追加于 CPU MVP 验收之后，基线为 `cde59eb`，开发分支 `dev/qwen-base-inference`。

- 修改 Runtime 后、添加新测试前，原有 **48 passed in 11.25s**。
- 新增输出解析/重试、失败原文保存、结构化 OOM/timeout、推理预算、Slurm 主机检查、Action 来源校验、真实子进程回收、冻结标签 oracle 与多库报告故障保留；最终本地 **69 passed in 20.40s**。
- Ruff check、format check、shell `bash -n`、git diff --check 通过。
- CPU `.venv` 的 `pip check` 仍通过；独立 GPU 环境 `pip check` 通过，Torch 2.2.2+cu121 / Transformers 4.46.3 在 CPU 上导入成功。没有在登录节点探测 GPU、加载权重或推理。
- 60 条新合成任务在原受限 SQLTool 上执行，并与独立 Python oracle 逐行一致；重建时校验冻结 manifest/database 指纹。

GPU 型号、显存、驱动、sm_70 编译架构、Singularity passthrough、Qwen 内容生成与真实 Base 报告均 **Pending**。未提交收费 GPU 作业，协议测试不能解释为模型验收通过。细节见 [M2A](milestone-2a.md) 与 [V100](v100-compatibility.md)。

首轮 M2A freecpuQ 作业 **20384068**（代码 `7a94940`）在 **node0928** 完成：free/normal、1 CPU/2 GB/5 分钟上限，无 GPU；**69 passed in 23.20s**，原 CPU smoke passed，Slurm **COMPLETED / 0:0 / 41s**。本地原始日志 `slurm-earl-cpu-smoke-20384068.out` 保留但不提交。随后为跨 SQLite 版本复现，将冻结任务清单改为 schema/数据内容指纹；轨迹文件字节哈希保持，并再次验证最终代码。

最终功能代码 **`2d64e27`** 的 freecpuQ 复验：作业 **20384095**，free/normal、**node0858**，**69 passed in 19.50s**，原 CPU smoke **passed**，Slurm **COMPLETED / ExitCode 0:0 / 37s**。本地日志 `slurm-earl-cpu-smoke-20384095.out` 保留但不提交。最终静态检查及 CPU/GPU 独立环境 pip check 通过；这里的 GPU 环境检查仅是 CPU 导入/依赖验证。

固定 Qwen tokenizer/chat template 的 CPU 检查见 `experiments/m2a-preparation/tokenizer-check.json`；180 prompt token、EOS 151645、assistant prefix 正确。完整权重缓存的 120 秒准备窗口超时（exit 124），保留未完成 blob，无完整缓存清单。没有加载模型，后续 GPU 作业前须继续完成缓存。
