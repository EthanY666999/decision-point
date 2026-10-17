# 运行维护说明

本轮目标：FrozenLake + Qwen2.5-0.5B-Instruct，GRPO/PaTR/Tree-Random/DAPO/ARPO 各 400 步、seed=42，各自独立初始化与最终 100×4 测试。方法和审计约束见 REPRODUCTION_AUDIT.md。

- 服务器 SSH alias `galaxy-a100`，用户 vipuser；项目 `/home/vipuser/patr-reproduction`。
- 本地项目 `/Users/ethan/Documents/ChatGPT/ACL2027/outputs/patr-reproduction`。
- 训练使用项目 `.venv/bin/python`，不要改 base Conda 或系统 CUDA。
- 单卡 A100 40GB；已下载固定版本 0.5B。运行不依赖外部 API 或持续网络下载。
- 验证阶段已结束，tmux `patr-async-validate` 已退出。`scripts/smoke_queue.sh` 完成五种异步调度 smoke 以及正式 GRPO 第 1 步的吞吐/检查点校验；`patr-formal` 中的 `scripts/start_after_validation.py` 已通过检查并启动正式队列。
- 正式队列 tmux 名称 `patr-formal`，命令见 README。`scripts/run_queue.py` 使用文件锁避免双队列，`--resume` 从步骤 1 检查点继续；已完成的方法跳过。
- 不要启动重复的 train.py。先看 tmux、进程、GPU、status/progress 文件与日志。
- `runs/engineering/` 保存同步批次调度的旧测试，旧大批次只做初始验证、步骤 0，没有正式训练数据。`runs/smoke/` 是当前调度的功能测试，不与正式结果混用。
- `scripts/cleanup_smoke_weights.py --apply` 只清除已完成 smoke 的大权重/优化器和已归档步骤 0 的工程检查点，保留原始记录；永不触碰 formal。避免无意义的 smoke 权重占满磁盘。
- 服务器自动报告由 queue 每分钟调用 `scripts/report.py`，输出 RESULTS.md / reports/summary.csv / reports/training.png。
- CUDA OOM 可以减少 micro-batch / token padding / vLLM 并发，记录 runtime_override；算法参数变更会被恢复检查拒绝。非有限 loss 或算法错误不允许无声忽略。
- 运行窗口到达会保存暂停；这不是云主机到期时间，不会自动续费。未完成的 400 步不能填最终效果表。
- 本聊天定时跟进 ID `patr-0-5b`，每 30 分钟一次。只在实质变化/故障/完成时通知。
- SSH 已配置本机专用密钥；不要将密码、密钥内容写进项目。临时 ControlMaster socket 断开时可用 `ssh -S none -o BatchMode=yes galaxy-a100`；避免连续高频新建 SSH 连接。

增量同步到本地时排除 `.venv/`、`artifacts/Qwen2.5-0.5B-Instruct/`、`checkpoint.*`、`final_model/`、缓存目录。保留报告、日志、源码、数据清单和原始压缩轨迹。最终模型与正式检查点留在远程，单独处理备份，不要误删。

2026-10-13 05:23：全部异步 smoke 完成，GRPO 正式步骤 1 用时约 362 秒，检查点已保存。gate 已通过；patr-formal 中 queue PID 17374 启动，GRPO 从 step 1 恢复。24 小时保护窗口起点是该队列启动时刻。之后仅补充了 resume 日志、状态展示与报告已完成步数处理，未改变数学计算或数据协议。以当前进程/文件为准，不盲用历史 PID。

2026-10-13 05:29：正式步骤 2 已完成，`resumed_from_step=1`，梯度和参数变化有限且非零，恢复验证通过。按前两步约 6 分钟/步估算，单方法 400 步约 40 小时；这是早期吞吐外推，后续随生成长度变化。五方法完整实验预计超过当前一天多的租期，未承诺一天完成，也没有缩短正式训练配置。
