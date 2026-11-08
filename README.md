# Decision Point：按模型规模分开的实验仓库

2026-11-08 整理。先选实验线，再进入对应分支；`main` 是总目录和历史证据入口。

| 实验线 | 入口 | 使用方式 |
| --- | --- | --- |
| **0.5B：历史实验与收尾证据** | [0.5B 索引](tracks/0.5b/README.md) | 已发布结果与未完成项分开标记，不因整理重跑 |
| **2B：当前 A100 80GB 实验** | [2B 索引](tracks/2b/README.md) | 沿用原运行分支、参数和上传目录 |
| **3B / 4B：合作者内网交付** | [交付说明](tracks/3b-4b/README.md) | 只 clone/pull `collab/3b-4b`，无需 push |


## 后续实验的独立教学数据

[难版 FLH Golden Trace — GPT-6.1 SOL](https://github.com/EthanY666999/decision-point/tree/data/flh-golden-trace-20261108)：1,000 条合格动作示范，1,021 次尝试，独立逐格与环境回放通过；排除全部 700 张验证/测试地图。数据包约 2.67 MB，含失败记录、教师调用证据、源码、哈希及复验脚本。

这是“完整动作计划后逐步回放”的 action-only 数据，尚未接入现有训练；不替换正在跑的 2B SFT 起点，也不改变 3B/4B 任务书。167 条完整对话超过 4,096 tokens，训练前须检查长度设置。固定数据提交：`c545848c4c52532683189d0c14bbd93beaab96c0`。

## 合作者直接用这条命令

```bash
git clone --depth 1 --single-branch --branch collab/3b-4b https://github.com/EthanY666999/decision-point.git ~/decision-point-3b4b
```

已有运行目录继续保留；不要在训练中切分支或 pull。后续更新、离线 ZIP 和内网留存见[更新说明](docs/COLLABORATOR_UPDATE.md)。
3B/4B 原任务书保留，新的研究方案尚未发布；这次整理不修改实验内容。

## 原有资料保留在哪里

| 路径 | 内容 |
| --- | --- |
| `code/`、`docs/` | 原版便携训练器和指南；**不是当前 2B 最新代码** |
| `results/`、`experiments/` | 0.5B 评估、Golden Trace 对照与研究记录 |
| `archives/` | UCF 扩展、无 SFT、源码与存储证据归档 |
| `history/` | V1–V4 等旧版、失败与纠错证据 |
| `golden-trace/` | 原 1,000 条 FrozenLake 教师示范，共享数据来源 |
| `golden-trace-swe/` | 独立 SWE 数据线，不是 FrozenLake 示范 |
| `tracks/` | 三条实验线的分支、代码、配置、结果定位 |

所有已有实验目录和结果保留原路径。旧 README/合作者说明原文另存于
[整理前入口快照](docs/archive/pre-organization-20261108/README.md)。
[分支与发布约定](docs/REPOSITORY_LAYOUT.md)明确每条线的写入范围；[机器可读索引](tracks/index.json)记录整理时提交。

这是论文方法的独立重实现，不是作者官方完整训练代码。不同模型、环境、SFT 与预算的结果必须分开报告。
不向 Mac 备份权重，不将权重、checkpoint、凭据加入 GitHub；历史失败证据也不删除。
