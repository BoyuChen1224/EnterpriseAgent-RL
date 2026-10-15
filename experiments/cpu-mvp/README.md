# CPU MVP 功能对照

运行日期：2026-10-15。最终报告由 `sbatch scripts/slurm/cpu_evaluate.sbatch` 在 freecpuQ 生成。

Job **20383968**，节点 **node0857**，1 CPU / 2 GB / 无 GPU，COMPLETED，ExitCode **0:0**，Elapsed **00:00:05**。这两组均为确定性 fixture，未使用 LLM，不能标为 Base/SFT/RL。

| 指标 | fixture-v1 | fixture-v1-recovery |
|---|---:|---:|
| test 任务成功 | 5/5 | 5/5 |
| SQL 可执行率 | 6/6 = 100% | 6/11 ≈ 54.55% |
| Tool Error Rate | 0% | 5/26 ≈ 19.23% |
| 平均工具调用 | 4.2 | 5.2 |
| 平均 episode latency | 0.1236 秒 | 0.1681 秒 |
| token 消耗 | 0（无模型） | 0（无模型） |

恢复策略在每个任务第一次 SQL 调用故意使用不存在的字段，再根据错误继续，因此具有更高错误率和成本；成功率相同只证明错误恢复链路正常。金额、排序、行语义和 Top-3 中间计数均由程序独立验证。latency 是小型合成库和该次节点环境下的实测值，包含工具子进程开销，不是模型吞吐。

相同 seed=17、数据库 SHA-256、5 个 task IDs、Settings 与 policy_kind；comparison 工具验证这些条件一致。费用按零单价配置计算为零，不含集群账单；这些值不能推广到真实模型。

- `fixture-report.json`：正常策略逐任务结果、配置、fingerprint、Slurm 元数据。
- `recovery-report.json`：错误恢复策略对应结果。
- `comparison.json`：同条件指标对照。

完整轨迹、SQL 审计、数据库和原始 Slurm 日志保留在本地 `artifacts/slurm-evaluation-20383968/` 和 Slurm 输出文件，因数据/日志 Git 排除策略不提交。开发期间登录节点的超时失败保留于 `artifacts/release/`，在 `docs/validation.md` 如实记录；此目录只收录随后计算节点完成的最终验收运行。

复现命令：从仓库根目录运行 `sbatch scripts/slurm/cpu_evaluate.sbatch`，会创建包含新 job ID 的独立 artifacts 目录，不覆盖本次原始报告。
