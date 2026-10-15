# 实施记录与交接

日期：2026-10-15，范围 Milestone 0–1。没有并行子 Agent，没有 GPU 作业，没有外部消息发送。

## 初始状态与版本管理

指定目录为空，按要求克隆 `https://github.com/BoyuChen1224/EnterpriseAgent-RL.git`；远端提示 empty repository。创建 `dev/cpu-agent-mvp` 分支，未覆盖已有文件或主分支。Git 阶段性记录使用本机已配置身份；功能代码提交为 `42bc006867c8654461b3feaf93b38e44edea7359`，测试/实验资料另作提交。最终 HEAD 与远程状态由 Git 历史和交付消息记录。

## 交付实现

采用明确的有界 Python runtime，环境只接收 PublicTask；没有为了目录齐全创建空训练模块。企业数据采用固定 seed 合成，5 张表、整数金额、历史单价、订单状态、空客户等业务边界。SQL 用只读 URI、authorizer 和单独 rlimit 子进程执行。

新增 plan/schema/dictionary/sql/check/finish 动作，实现错误恢复、预算耗尽、超时、证据绑定结束。轨迹 v1.0 包含逐步输入输出、策略版本、配置、数据库指纹、成本、终局与奖励；支持 JSON/JSONL 和确定性 action replay。

奖励独立于策略，最后用参考 SQL 执行验证，不将答案返还给策略探测。评测分别记录成功率、SQL 执行率和效率，fixture 类型显式标注；增加真实本地模型 HTTP 协议接口及 Slurm/loopback 限制，但没有运行真实模型。

FastAPI 采用本地单进程、容量上限 8、持久任务状态、轨迹查询、版本选择和比较请求；批量预留容量避免部分提交泄漏。自定义任务不自动冒充已验证 benchmark。CLI、安装脚本、CPU smoke、免费 Slurm、GPU metadata 预案与 CI 一起落地。

## 已遇问题与处理

- bwrap 在集群无法创建用户命名空间：使用经审批的项目范围命令，没有更改系统配置。
- 默认 pip 镜像的环境代理握手超时：对单次命令取消代理，使用官方 PyPI，依赖安装在 `.venv`。
- SQLite `file:` URI 初次连接遗漏 `uri=True`：CPU smoke 捕获并修复。
- COUNT(*)/部分索引扫描出现 database=None、空列名 READ 授权回调：按实测特殊情况修复，仍限制固定业务表，非法 SQL 测试保持通过。
- 新解析到的 AnyIO 导致 Starlette 测试弃用警告：固定 AnyIO 4.8.0，锁定完整传递依赖。
- 最终登录节点覆盖率回归曾出现 45 通过、2 失败；一次同步 Schema 读取耗时 46.6 秒并触发 episode_timeout，最新恢复对照出现 4/5。保留失败轨迹，随后把 Schema 读取也改为有硬超时的受限子进程，并转到免费计算节点完成最终验收，不将该次失败归为通过。
- GitHub REST API 限流：用临时 partial clone 的 Git 对象读取真实源码和 SHA。
- 历史 verl 仍含 FA2/BF16 默认路径：不直接装到 CPU 环境，不宣称 V100 兼容。
- quota/stale NFS 无法给出可靠个人配额：记录未确认，后续下载模型前检查。

## 交接入口

先读 README、architecture、environment-audit、validation，再读 roadmap 和 upstream-research。使用 `bash scripts/setup/bootstrap.sh`、`pytest -q`、`python scripts/evaluation/cpu_smoke.py` 复现。数据初始化保护已有文件；测试使用临时数据库，不要删除用户数据以“重置环境”。

`artifacts/` 内存在本次实际评测报告、轨迹、SQL 审计和示例数据库，未提交 Git；提交的 `experiments/cpu-mvp/` 仅保留小型聚合报告。Slurm 原始日志被忽略，验收文档保存 job ID、节点、退出码及结果。新增模型数据时继续遵守同样的 Git 排除规则。
