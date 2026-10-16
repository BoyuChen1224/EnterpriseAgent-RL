# 集群环境审计

检查日期：2026-10-15（Asia/Shanghai）。范围：Milestone 0–1，登录节点只读元数据和轻量 CPU 验收；没有提交 GPU 作业、下载模型或执行训练。

## 实测环境

| 项目 | 实际检查结果 | 工程含义 |
|---|---|---|
| 初始目录 | 指定工作目录为空，没有 `.git` | 从用户指定 origin 克隆；远端为空仓库，没有覆盖已有代码 |
| OS / kernel | CentOS Linux 7；3.10.0-862.el7.x86_64；登录节点 ln02 | 不假设现代 glibc 或用户命名空间 |
| glibc | 2.17 | CPU 原生扩展实际使用兼容 wheel；现代 GPU wheel 需单独检查 |
| 默认编译器 | GCC/G++ 4.8.5 | 不直接拿默认 GCC 编译现代训练算子 |
| 可用编译器 module | GNU/gcc 6.1、9.3、10.1、12.2、14.2 | 按 CUDA 兼容矩阵在作业内选择，不修改系统 |
| Python | 用户 Miniconda Python 3.12.7 | 在仓库 `.venv` 隔离安装，不修改已有 conda 环境 |
| CPU 包环境 | FastAPI 0.115.6、Pydantic 2.10.6、pytest 8.3.4、Ruff 0.9.10 | 完整传递依赖固定在 `requirements/cpu-py312-linux.lock` |
| SQLite | 3.45.3 | 支持 authorizer、progress handler、`setlimit`，测试通过 |
| 既有基础环境 Torch | 2.2.0+cu121，编译 CUDA 12.1；CPU import 成功 | 仅检查版本；未在项目 `.venv` 安装 Torch，未验证设备侧可用性 |
| 登录节点 CUDA | `nvcc`、`nvidia-smi` 默认 PATH 均无 | 不把它解释为计算节点无驱动；没有在登录节点加载模型 |
| CUDA module | 9.1、10.0/10.1/10.2、11.4.3、11.8.0、12.1.0、12.3.1 | 已读取 11.8 module 配置；不是 GPU 驱动/运行验证 |
| Slurm | 21.08.8-2；sinfo/scontrol/sbatch/srun 可用 | 已实际提交并完成免费 CPU 作业 |
| 容器 | Docker、Podman、Apptainer 默认 PATH 无；module load singularity/3.11.2 后 `singularity-ce version 3.11.2` | Singularity 可执行文件可用；未验证镜像运行和 GPU passthrough |
| 命令沙箱 | bwrap 创建用户命名空间失败 | 本次通过审批的命令执行完成，不修改内核或系统设置 |
| 存储 | `/public` 共享总容量约 2.1 PB，可用约 340 TB；`/tmp` 所在本地文件系统可用约 25 GB | 共享可用量不是个人配额；未全盘扫描用户目录 |
| 用户配额 | `quota -s` 对历史 NFS 挂载报 stale file handle，未获得可靠配额 | 后续下载权重前需另行确认个人 quota |
| locale | C.UTF-8 在当前登录 shell 报不可用警告 | 不影响本次 UTF-8 Python/JSON 测试；未修改系统 locale |

## Slurm 分区与设备信息边界

`sinfo -o '%P %a %l %D %G'` 与 `scontrol show node` 实际看到：

| 分区 | 节点数 | 单节点 GRES | 抽样节点 | 抽样节点内存 |
|---|---:|---|---|---|
| freecpuQ | 39 | 无 GPU | CPU 验收在 node0933 | 本次仅申请 2 GB |
| gpu2Q | 10 | gpu:tesla:2 | gpu201 | RealMemory=384000 MB |
| gpu4Q | 9 | gpu:tesla:4 | gpu401 | RealMemory=384000 MB |
| gpu8Q | 4 | gpu:tesla:8 | gpu801 | RealMemory=384000 MB |

Slurm 标签只给出 `tesla`，**不能由此实证 V100/V100S 型号、单卡显存、驱动版本、Compute Capability 或 NVLink 拓扑**。用户提供了 Volta/CC 7.0 的硬件说明，本次将其作为后续兼容设计约束，设备实测留到后续授权的短 GPU 作业。准备了 `scripts/slurm/gpu_inventory.sbatch`，本次没有提交。

分区配置显示 MaxTime=UNLIMITED；这不代表账户/QOS/使用政策没有限制。仍遵守用户给定的 freecpuQ 最长三天、最多 48 核限制，本次验收主动限制为 1 核、5 分钟。账户/QOS 关联已只读确认存在 free/normal；GPU 账户、QOS 与费用应在下一阶段按授权选定，不能因为队列显示 up 就推断可免费使用。

## 安装与实际验证

默认 pip 镜像通过环境代理访问超时。仅在安装命令中取消代理变量并改用官方 `https://pypi.org/simple` 后成功；未改动系统配置，未输出代理凭据。

CPU 固定依赖、editable 安装、`pip check`、pytest、CLI 完整评测和免费 CPU Slurm 验收都已执行。pydantic-core 2.27.2 与 Ruff 0.9.10 使用 manylinux_2_17 x86_64 wheel，在当前 glibc 2.17 上实际运行成功。完整验收结果见 `docs/validation.md`。

## GPU 依赖决策

CPU runtime 不依赖 Torch、Transformers、LangGraph 或任何训练框架。后续候选优先复用可导入的 Torch 2.2 系列环境并在独立训练环境测试 Transformers 4.46.3，而非直接升级当前环境；这些只是候选，不是已经锁定或验证的训练组合。

Qwen 官方模型卡说明 Transformers <4.37 无法识别 qwen2，且示例使用 `torch_dtype='auto'`。本项目后续必须显式指定 FP16，不能根据权重默认精度推断 Volta 支持。来源：[Qwen2.5-1.5B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)。

上一阶段曾将 V100 路径暂定为 Transformers `attn_implementation='eager'`，Job 20385898 已证明 FP16 eager 存在 QK overflow；当前生产路径改为 `attention_backend='sdpa_math'`，并由 Runtime 局部强制 SDPA math。FlashAttention-2 的 CUDA 路径对 GPU 架构有要求，不能默认用于 Volta；见 [官方项目](https://github.com/Dao-AILab/flash-attention)。同时本次检查了 verl v0.4.1 源码中的 FlashAttention-2/BF16 默认值，因此不直接安装其完整 requirements，详见 `docs/upstream-research.md`。

下一阶段先采集真实驱动、显存、CC、编译架构和容器运行情况，再决定 CUDA wheel、Python 版本、编译器、SFT 与 GRPO 后端。当前没有任何 GPU 显存、吞吐、训练稳定性或能力提升实测数据。
