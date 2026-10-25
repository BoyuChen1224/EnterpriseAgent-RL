# M2A 准备记录

2026-10-15，完成 CPU 工程与 tokenizer 检查，当时尚未运行 GPU 模型。

| 检查 | 结果 |
| --- | --- |
| 本地 CPU 回归 | 69 passed |
| Job 20384068 / node0928 | 69 passed，smoke 通过，0:0，41 s |
| Job 20384095 / node0858 | 69 passed，smoke 通过，0:0，37 s |
| 合成基准 | 60 条 reference SQL 与独立 Python 结果一致 |
| tokenizer | 180 prompt token，EOS 151645，assistant prefix 正确 |
| 当时缓存 | 120 s 下载窗口超时，保留 incomplete blob |

[Tokenizer 记录](tokenizer-check.json) 是当时的 CPU 结果。缓存续传在 [M2B](../m2b-preparation/README.md) 完成；GPU 后续结果见 [模型成绩](../../docs/results.md)。
