# 2B：当前 A100 80GB 工作线

模型：**Qwen3.5-2B**。2026-11-08 已与 `ucf 2b` 聊天核对；本索引不修改正在运行的代码、模型参数或队列。

| 用途 | GitHub分支 | 服务器工作树 |
| --- | --- | --- |
| 原版 FrozenLake / F02 | `codex/q35-2b-20261107` | `/data/decision-point` |
| 后续 FLH / Sokoban 队列 | `codex/q35-hard-env-20261107` | `/home/vipuser/decision-point-q35-hard` |

- [原版代码与配置](https://github.com/EthanY666999/decision-point/tree/codex/q35-2b-20261107/code)
- [难环境代码与配置](https://github.com/EthanY666999/decision-point/tree/codex/q35-hard-env-20261107/code)
- [原任务书](https://github.com/EthanY666999/decision-point/blob/codex/q35-2b-20261107/docs/q35_2b_taskbook_20261107.md)
- [难环境修订](https://github.com/EthanY666999/decision-point/blob/codex/q35-hard-env-20261107/docs/q35_hard_revision_20261107.md)
- [原版已发布结果](https://github.com/EthanY666999/decision-point/tree/codex/q35-2b-20261107/results/q35_2b)
- [难环境分支结果目录](https://github.com/EthanY666999/decision-point/tree/codex/q35-hard-env-20261107/results)

## 发布与并发约定

持续上传路径仍为 `results/q35_2b/`；已有难环境数据放在其原发布目录。
由 `ucf 2b` 聊天通过增量 bundle 和独立发布仓库上传；本次核验没有独立自动上传守护进程。
`q35_report_queue.py --watch` 只生成远端报告，不能把报告更新当作已经 push。
后续 GitHub 内容可能晚于服务器，判断训练进度须读取实际运行状态。

两个分支只由该工作线维护；整理者不提交、不强推、不改路径，也不共享其发布 bare 仓库或锁。
本轮保留训练工作树、模型路径、数据哈希、实际 run config、source snapshot 和评估口径。
每次发布应同时记录 run ID、来源提交、参数配置、步数及验证/最终测试身份；原版与 FLH/Sokoban 不混成同一组结果。
分支页面展示当前版本；复核某次实验须用其记录的固定提交，不能用最新 HEAD 代替。

## 后续实验的独立教学数据

[难版 FLH Golden Trace — GPT-6.1 SOL](https://github.com/EthanY666999/decision-point/tree/data/flh-golden-trace-20261108)：1,000 条合格动作示范，1,021 次尝试，独立逐格与环境回放通过；排除全部 700 张验证/测试地图。数据包约 2.67 MB，含失败记录、教师调用证据、源码、哈希及复验脚本。

这是“完整动作计划后逐步回放”的 action-only 数据，尚未接入现有训练；不替换正在跑的 2B SFT 起点，也不改变 3B/4B 任务书。167 条完整对话超过 4,096 tokens，训练前须检查长度设置。固定数据提交：`c545848c4c52532683189d0c14bbd93beaab96c0`。
