# 本地模型服务

`model_server.py` 提供模型推理 HTTP，`serving.py` 提供任务 API。`run_sft_n_api.py` 同时管理两者的启动、就绪检查和回收。

现成 profile 是 `configs/serving/sft-n.toml`，绑定 **Qwen2.5-1.5B-Instruct / sft-n-step27**。Qwen3 已跑过模型 HTTP 与 Runtime 评测，其 Agent API profile 仍待接入。

## 启动

先准备 profile 指定的模型缓存、Adapter、train 任务和数据库。在 Slurm 分配的 GPU 节点，从仓库根目录运行：

~~~bash
.venv-gpu/bin/python scripts/serving/run_sft_n_api.py \
  --repo-root "$PWD" \
  --profile "$PWD/configs/serving/sft-n.toml" \
  --mode serve --model-port 8001 --agent-port 8000
~~~

已有 Slurm wrapper：

~~~bash
EARL_MODE=serve sbatch --partition=gpu4RQ scripts/slurm/sft_n_api.sbatch
~~~

wrapper 使用 1 GPU、2 CPU、16 GB、25 分钟，账户 `pi_daipeishan`、QoS `gpuq`。按集群实际情况选择兼容分区。参数可通过 `--help` 查看。

## 模式

| 模式 | 行为 |
| --- | --- |
| `--mode smoke` | 启动服务，执行固定的两项 train smoke，保存结果后退出 |
| `--mode serve` | 保持服务，等待客户端任务 |
| `--no-smoke` | 检查就绪后退出；与默认 smoke 模式配合使用 |

未指定 `--run-dir` 时创建唯一目录 `artifacts/service-runs/run-<时间>-<nonce>/`。指定的已有目录会被拒绝。

全局 deadline 覆盖启动、请求、等待和清理。TERM/INT 回收自身子进程组；退出码分别为 143/130。超时或子服务故障返回非零，已完成任务的结果保留。

## 请求

以下示例假设 Agent API 监听 8000：

~~~bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/models
curl -s http://127.0.0.1:8000/benchmarks

curl -s -X POST http://127.0.0.1:8000/tasks \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"先使用 schema 确认客户表字段，列出客户名称和区域编号，返回 customer, region_id，按客户名排序。","benchmark_task_id":"train-1-schema"}'

curl -s "http://127.0.0.1:8000/tasks/<job_id>"
curl -s "http://127.0.0.1:8000/tasks/<job_id>/trajectory"
~~~

把 `<job_id>` 换成提交接口返回的 ID。公开接口返回 task_id、prompt、模型身份和任务状态。参考答案只用于独立评分，不进入模型请求。

## 运行记录

run 目录保存 `commands.json`、进程状态、模型事件、轨迹、SQL audit、逐任务结果、summary 和 manifest。

计数优先读取原始事件或 audit，缺失才使用轨迹；两者互斥，缺来源记 null。GPU allocation 和服务 wall time 分开统计。

V100 实跑 Job `20419673` 完成了两项 smoke：`train-1-empty` 成功，`train-1-schema` 耗尽动作预算；两项技术记录均完整。生命周期、信号和部分结果保存另有 CPU 回归。

当前服务绑定本地地址，尚未实现公网认证、多租户、外部队列和跨 Job 自动累计预算。模型训练与任务评测入口见 [复现说明](reproducibility.md)。
