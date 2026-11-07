# 3B / 4B 合作者交付

**当前分支：`collab/3b-4b`。只拉取这一条分支即可；不需要 push。**

2026-11-08 从 main 的 `fd573a6c08941b0260a7ffc260a6903257eb7cb1` 隔离交付。本次不修改训练实现、模型参数或原实验清单；新方案尚未发布。

1. 先读[更新与内网使用说明](docs/COLLABORATOR_UPDATE.md)。
2. 按[保留的3B/4B任务书](docs/PATR_3B_4B_REPRODUCTION_TASK_2026-11-01.md)核对已完成项；本次整理不触发重跑。
3. 在无活跃训练的目录更新；记录 `git rev-parse HEAD`，执行 `python3 code/scripts/verify_release.py`。

| 路径 | 内容 |
| --- | --- |
| code/ | 原便携训练、SFT、评估、配置生成及测试；未改算法 |
| golden-trace/ | 同一份1,000条FrozenLake示范及校验，不是SWE |
| results/supplemental500/ | 原补500评估器读取的协议、题表、五批校验数据；不是新实验成绩 |
| docs/ | 当前入口、保留任务书、历史来源说明 |
| TRACK.json / RELEASE_MANIFEST.json | 交付来源与逐文件校验 |

该分支不携带0.5B大批历史归档或2B运行结果；完整历史见[main总目录](https://github.com/EthanY666999/decision-point)。
公共模板 `code/configs/frozenlake_05b.json` 仍是3B/4B配置生成器的依赖，不能按文件名删除。
Golden Trace原始说明保留其发布日期；历史结果链接可到main查看，不作为新待办。
模型权重与Python/CUDA依赖不包含在本仓库。CPU校验通过不代表已完成目标GPU验收。
