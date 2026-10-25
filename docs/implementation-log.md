# 开发记录

| 阶段 | 主要工作 | 结果 |
| --- | --- | --- |
| CPU MVP | Runtime、只读 SQL、轨迹、独立评分、API | fixture 和错误恢复测试通过 |
| 模型接入 | Transformers 服务、冻结任务、V100 诊断 | FP16 + SDPA math 路径稳定 |
| 首轮 1.5B SFT | 20 条 train 轨迹、15 次更新 | loss 下降，validation 仍 0/20 |
| Runtime 对齐 | 逐决策监督、输入和错误反馈诊断 | 后续冻结 1.5B SFT 为 8/20 |
| 1.5B GRPO | 在线采样、更新、恢复、三臂评测 | SFT/GRPO 均 8/20 |
| 本地服务 | 模型 HTTP、任务 API、进程生命周期 | 两项 V100 train smoke 完成 |
| Qwen3-8B | 54 条决策 SFT、分组 rollout、两次 GRPO 更新 | 0/20 → 13/20 → 13/20 |
| Git 与 CI | main、阶段提交、独立 CPU 测试 | 197 passed，两个历史产物检查 skip |

## 主要修复

- SQLite 只读 URI、授权回调和 Schema 查询超时：使用真实子进程检查。
- 1.5B FP16 eager 的 QK 溢出：切换局部 SDPA math。
- 手工 LoRA 参数名与选择器不匹配、checkpoint 输入未启用梯度：增加参数和 backward 检查。
- checkpoint 反向重算超出 math context：扩展作用域并验证恢复等价。
- 请求已发出但响应未保存、wrapper 返回码未收尾：分别修复持久化和阶段调度。
- CPU CI 依赖缺失、本地路径与 ignored 数据：独立 CPU 锁文件、临时 fixture、显式历史产物测试。

早期失败、原始日志与修复后的新 run 分开保存。详细历史见 [测试](validation.md)、[首轮 SFT](experiments/sft-pilot-report.md) 和 [修正训练](experiments/sft-pilot-corrected-report.md)。

源码按阶段提交到 main。模型结果见 [结果](results.md)，运行方法见 [复现](reproducibility.md)。
