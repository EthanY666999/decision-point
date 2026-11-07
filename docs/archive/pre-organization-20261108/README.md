# PaTR / Golden Trace：FrozenLake 实验与复现

**2026-11-03：合作者只使用 `main`。** SFT、RL、评估、Golden Trace 数据和任务书现已在同一分支；无需再拉实验分支，也不需要手工 `sed` 修改训练器。

## 给合作者的唯一任务入口

**[3B / 4B 精简复现任务书](docs/PATR_3B_4B_REPRODUCTION_TASK_2026-11-01.md)**：单张 A100 80GB 串行，优先完成两个模型的 FrozenLake 主对照与 SFT 对照。取消朋友这条线的 SWE、K 扫描、CERT、DAPO/ARPO 等附加任务；额外 seed 与 SFT→GRPO8 暂不排队。已完成的不用重跑，正在正常运行的先完成。

```bash
git clone --depth 1 --single-branch --branch main https://github.com/EthanY666999/decision-point.git ~/patr-ws0
cd ~/patr-ws0
```

已经使用 main 且没有本地代码修改：在训练全部退出后执行 `git pull --ff-only origin main`。从旧分支迁移、有手工修改或需要离线导入：见 **[内网与单分支更新说明](docs/COLLABORATOR_UPDATE.md)**。不要在正在训练的目录切换代码。

内网机器只需拉取，不要求 push、不要求提供 GitHub 写权限，也不要求把日志、轨迹或权重传出来。允许对外交流时填写任务书的简短汇总即可；完整证据留在内部。

## 仓库内容

| 目录 | 用途 |
| --- | --- |
| `code/` | 当前便携训练、SFT、评估和本地统计入口 |
| `golden-trace/` | 1,000 条 FrozenLake 教师轨迹及逐条回放校验；不是 SWE 数据 |
| `docs/` | 合作者任务书、更新说明、原 0.5B 指南、方法差异与发布验收 |
| `results/` | 已完成历史实验的评估与证据，不是朋友需要重复执行的任务列表 |
| `experiments/` | 原实验分支合入的研究协议与执行记录；具体服务器路径和旧排程仅供追溯 |
| `history/` | 更早版本的精简历史；不作为当前运行入口 |

这是论文方法的**独立重实现**，不是作者官方完整训练代码。Qwen2.5-3B FrozenLake 可参照论文表 1；Qwen3-4B FrozenLake 是扩展实验。额外 SFT、优化器差异、题集和响应长度口径必须披露，不能用数值接近宣称完整复现。

## 配套说明

- [原 0.5B 复现指南](docs/REPRODUCE.md)
- [方法、来源和已知限制](docs/METHOD_AND_LIMITATIONS.md)
- [本次合并与工程变更](docs/COLLABORATOR_RELEASE_2026-11-03.md)
- [历史发布验收](docs/VALIDATION.md)
- [Golden Trace 数据来源与校验](golden-trace/README.md)
- [历史结果索引](results/README.md)

不提供 GPU 模型权重、虚拟环境、下载缓存或账号信息。模型可从内部镜像/已有目录加载；模型与 Python 依赖是代码仓库以外的输入，一次克隆不代表它们已安装。

维护约定：合作者文档、运行所需脚本和配置入口随 main 一起发布；不再要求合作者切换研究分支。旧分支保留已有链接与历史，不作为新的交付入口。更新任务范围不等于修改已经完成实验的历史记录。
