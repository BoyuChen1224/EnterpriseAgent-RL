# Milestone 2A：V100 兼容性与证据

检查日期：2026-10-15（Asia/Shanghai）。起点 `dev/cpu-agent-mvp@cde59eb`；开发分支 `dev/qwen-base-inference`。

**GPU 验收 Pending。用户尚未授权收费 GPU 作业，本阶段没有提交 gpu2Q 作业，没有加载 Qwen，没有真实模型推理或训练结果。** 下表中的 Pending 不能视为通过。

## 已实测的登录节点元数据

| 项目 | 结果 | 证据/边界 |
|---|---|---|
| Slurm 用户关联 | `pi_daipeishan_r`: normal/overdue；`pi_daipeishan`: cpuq/fatq/gpuq/normal；`free`: normal | `sacctmgr -n -P show assoc where user=$USER ...`；关联存在不等于收费授权 |
| gpu2Q | `gpu[201-210]`，允许 QOS `gpuq,normal`，MaxTime=UNLIMITED；BillingWeights `cpu=1.0,mem=0.25G,gres/gpu:tesla=2.0` | `scontrol show partition gpu2Q`；不由 tesla 标签推断 V100 型号或账单价格 |
| QOS gpuq | MaxJobsPU=200，MaxSubmitPU=50；MaxWall/MaxTRESPU 输出为空 | `sacctmgr show qos` 原始字段；未验证账户/分区交集和调度可接受性 |
| QOS normal | MaxJobsPU=200，MaxSubmitPU=5，MaxWall=3-00:00:00，MaxTRESPU=cpu=288 | 不是用户可长期占满资源的许可；脚本主动限制单卡和短时限 |
| Lustre 个人用量 | 2,579,951,548 KiB，约 2.40 TiB | `lfs quota -u "$USER" /public` |
| Lustre 软/硬配额 | 3,221,225,472 / 4,294,967,296 KiB，即 3 / 4 TiB；软配额剩余约 613 GiB | 这是用户级配额；`df` 的共享剩余空间不能替代它 |
| 旧 quota 命令 | `quota -s` 对历史 NFS 挂载报 stale file handle | 使用 Lustre 查询补齐；没有修改挂载 |
| 模型缓存位置 | 默认项目 `artifacts/model-cache`，可用 `EARL_MODEL_CACHE` 显式指定 | 未扫描或复用其他项目模型；离线服务不自行下载 |

独立 GPU 环境已实际安装；`pip check` 通过。登录节点仅完成 Torch 2.2.2+cu121、CUDA 编译版本 12.1、CPU tensor 及 Transformers 类导入检查；没有初始化 GPU 或加载模型。这证明当前 glibc 下的 CPU 导入兼容性，不证明计算节点驱动或 FP16 推理兼容性。

## 必须由获准 GPU 作业补齐的证据

| 验收项 | 当前状态 | 预期证据 |
|---|---|---|
| 实际型号/总显存/驱动/Compute Capability | Pending | `artifacts/gpu-inventory-<job>/inventory.json` 的 nvidia-smi、Torch properties |
| CUDA driver API / runtime / nvcc | Pending | 审计分别记录三者，区分 wheel 自带 runtime 与 module toolkit |
| Torch 编译架构及 sm_70 | Pending | 审计 get_arch_list；服务启动也检查 sm_70 |
| Singularity `--nv` | Pending | 用已有本地镜像指定 `EARL_AUDIT_IMAGE` 后运行；仅 module 可用不算通过 |
| Qwen FP16/eager 加载与生成 | Pending | `server/model-load.json`、`single-generation.json`、原始 generations.jsonl |
| Agent 自主多轮与业务成功 | Pending | 原有 top-2025 的模型轨迹、SQL 工具反馈和独立 reward；模型加载成功不能代替它 |
| Base validation/test 评测 | Pending | 完整 report.json、每任务轨迹、冻结 manifest 哈希、执行配置 |
| 加载耗时、峰值显存、吞吐、延迟 | Pending | 每次生成和加载的真实测量；当前没有数值 |

## 固定兼容路径

`configs/model/qwen2.5-1.5b.toml` 固定模型为 `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。已通过官方固定 revision 页面核验：[模型文件](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)。目标是该 Instruct 检查点的未进行项目 SFT/GRPO 的基线；不把它称为纯预训练 Qwen2.5-1.5B。

采用独立 `.venv-gpu`、Torch 2.2.2（CUDA 12.1 wheel）、Transformers 4.46.3、NumPy 1.26.4、tokenizers 0.20.3、safetensors 0.4.5。完整版本见 `requirements/gpu-py312-linux.lock`；CPU `.venv` 与原锁文件保留。CUDA 12.1 wheel 的候选选择参考 [PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/)。驱动能否运行该组合仍须计算节点实测；若不兼容，先按审计结果建立另一份锁，不升级系统驱动。

上游模型配置标为 BF16，服务不使用 `torch_dtype="auto"`，而是显式 `torch.float16`；attention 固定 eager，加载后校验实际 dtype 与 attention。Qwen2 已包含于固定 Transformers 版本，见 [官方 Qwen2 文档](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/qwen2)。没有启用 BF16、FlashAttention-2、vLLM、verl 或自定义新架构算子，也没有做训练。以上是实现选择，不是 V100 通过结论。

## 授权后的运行顺序

先准备依赖与缓存（不加载模型），再经授权按审计→单轮→Agent smoke→validation→冻结配置后的 test 顺序提交。具体命令和产物见 `docs/milestone-2a.md`。本次只做 shell 语法检查，没有执行 `sbatch --test-only` 或真实 GPU 提交；脚本资源声明不等于调度器已接受账户/QOS。
