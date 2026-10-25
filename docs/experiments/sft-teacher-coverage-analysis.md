# Train 教师覆盖扩展

历史 1.5B SFT 数据构建。输入为 20 个 train 任务、两个 train 数据库，不将 validation/test 作为训练监督。

## 原有覆盖

确定性策略成功 6/20，覆盖 region、top、inactive 三类模板。其余 14 项中，12 项模板不支持，2 项约束结果错误。

20 条 train reference SQL 均可通过受限 SQLTool 执行，并得到独立评分。因此用 reference-SQL teacher 补齐未覆盖任务。

## 生成方式

`reference_sql_teacher.py` 拒绝非 train 任务。它把参考 SQL 作为 train 教师的 Action，走与普通策略相同的环境：

~~~text
公开任务 → Plan / Schema / Dictionary → SQLTool
        → 实际 observation → Check → Finish → 独立 Verifier
~~~

Top-3 先执行订单数 probe；recovery 先执行错误查询，再读取实际反馈并修复。最终答案由实际 SQL 结果提取。

保留 6 条原确定性成功轨迹，另生成 14 条 `programmatic_reference_sql_teacher` 轨迹。来源、任务、数据库、轨迹哈希与评分随样本保存。

## 数据结果

| 项目 | 数量 |
| --- | ---: |
| train 任务 / 成功轨迹 | 20 / 20 |
| 确定性教师 / reference-SQL 教师 | 6 / 14 |
| 模板覆盖 | 10/10 |
| 重复轨迹 | 0 |
| 评分与 reward 一致 | 20/20 |

输出位于 `artifacts/sft-data-train-teacher-v2/`，包含 dataset、metadata、quality report、source manifest 和原始轨迹。

训练样本不含私有 `reference_sql` 字段或 Verifier 答案。SQL 监督来自教师实际执行的 assistant Action，不能将教师成绩计入 Base 模型成绩。

这批数据只有 20 条整轨迹，随后用于 [早期 SFT](sft-pilot-corrected-report.md)；当前 Qwen3 使用的是后续 54 条逐决策数据，见 [训练说明](../qwen3-post-training.md)。
