# M2B：缓存与训练来源准备

历史阶段，2026-10-16。该阶段完成 1.5B 缓存、CPU 回归与 train 数据审计，尚无 GPU 生成结果。

## 模型缓存

固定 `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。

缓存核对通过：8 个文件，无 incomplete blob；权重 3,087,467,144 bytes，唯一物理缓存共 3,098,967,011 bytes。权重 SHA-256：

~~~text
dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee
~~~

下载接续了 M2A 的 incomplete blob。模型服务使用 `local_files_only=True`，完整性检查后离线加载。

[缓存记录](../experiments/m2b-preparation/qwen-cache-verification.json) · [冻结配置](../experiments/m2b-preparation/experiment-freeze.json)

## Train 数据审计

确定性策略运行 20 项 train 任务，6 项成功、14 项失败：12 项模板不支持，2 项约束结果错误；覆盖 3/10 个模板。它们按 `deterministic_fixture_expert` 记录。

这些轨迹不属于 Base 模型结果。随后使用 reference-SQL teacher 补齐其余 14 项，见 [教师覆盖](experiments/sft-teacher-coverage-analysis.md)。

## 阶段检查

CPU 回归 69 passed。缓存、训练来源和配置摘要保留在 [准备目录](../experiments/m2b-preparation/README.md)。

后续模型运行按加载、协议、Agent 任务、validation 分层检查。首轮真实 Base 结果见 [Benchmark](experiments/base-agent-benchmark.md)，当前两套模型结果见 [汇总](results.md)。
