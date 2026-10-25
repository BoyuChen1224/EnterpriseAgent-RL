# M2A：模型接口与业务基准

历史阶段，2026-10-15。此时完成 CPU 工程，尚未运行 GPU 模型；后续实跑见 [V100](v100-compatibility.md) 和 [结果](results.md)。

## 实现

- Transformers loopback 服务，固定模型 revision，离线读取已校验 cache。
- Infrastructure、Protocol、Agent 三层 smoke，分别检查加载、Action 格式和业务任务。
- 原始请求/输出、重试、token、耗时、显存和结构化错误保存。
- runtime、SQL、Verifier 和报告字段接入真实模型 Policy。

入口为 `scripts/inference/run_qwen.py`、`scripts/evaluation/base_agent_benchmark.py` 和相应 Slurm wrapper。

## 冻结基准

`benchmarks/m2a/manifest.json` 包含 60 项任务、六个数据库实例，train/validation/test 各 20 项。年份分别为 2021/2023/2025，seed 和实体名称分开，仍共享五表 schema 与部分模板。

任务覆盖 Schema、区域收入、Top-3、客户、商品、空结果、聚合约束和错误恢复。Top-3 另要求独立的 `paid_orders` 查询，恢复任务要求出现真实 SQL 错误后修正。

60 条 reference SQL 均通过受限 SQLTool，并与独立 Python 聚合结果核对。清单使用 schema/行内容指纹，轨迹回放另使用 SQLite 文件字节哈希。

## 阶段检查

CPU 回归由 48 项扩展到 69 项。最终 freecpuQ Job `20384095`：node0858、69 passed、smoke passed、0:0、37 s。

该阶段只取得 tokenizer/配置，120 s 权重下载超时，保留 incomplete blob。完整 cache 在 [M2B](milestone-2b.md) 完成。

当时的依赖和检查摘要保留在 [准备目录](../experiments/m2a-preparation/README.md)。当前运行方式见 [复现](reproducibility.md)。
