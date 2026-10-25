# 环境

## 已使用的环境

| 用途 | Python | Torch | Transformers | 设备 |
| --- | --- | --- | --- | --- |
| CPU Runtime / CI | 3.12 | 2.5.1+cpu 用于后训练测试 | 4.51.3 用于随机小模型测试 | CPU |
| Qwen2.5-1.5B | 3.12.7 | 2.2.2+cu121 | 4.46.3 | V100 / V100S 32GB |
| Qwen3-8B | 3.12.7 | 2.5.1+cu121 | 4.51.3 | V100 / V100S 32GB |

依赖分别记录在 `requirements/cpu-py312-linux.lock`、`cpu-training-py312-linux.lock`、`gpu-py312-linux.lock` 和 `qwen3-py312-linux.lock`。

GPU 路径使用 FP16 和 SDPA math。Qwen3 关闭 thinking。环境安装与入口见 [复现](reproducibility.md)。

## 集群记录

最初环境检查于 2026-10-15 完成：CentOS 7、glibc 2.17、Python 3.12.7、SQLite 3.45.3、Slurm 21.08.8-2。后续计算节点实际运行确认了 V100/V100S 32GB、CC 7.0。

已用过的 GPU 分区为 `gpu2Q`、`gpu4Q`、`gpu4RQ`、`gpu8Q`，账户 `pi_daipeishan`、QoS `gpuq`。分区名区分大小写，资源与排队状态以提交时为准。

Singularity 3.11.2 曾在登录节点检查可执行，GPU passthrough 没有在该次检查中验证。模型训练采用独立 venv，不依赖容器。

## 安装和存储

CPU 与 GPU 依赖分开安装。模型下载到本地 cache，校验 revision、文件哈希和完整性后离线加载。权重、数据库、checkpoint 和日志由 `.gitignore` 排除。

初期遇到过代理超时、缓存下载中断、quota 命令读取失效 NFS 挂载等问题。后续完成了固定 snapshot 的下载与校验；早期状态见 [M2B](milestone-2b.md)，当前 GPU 数值路径见 [V100](v100-compatibility.md)。
