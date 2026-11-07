# 3B / 4B：内网合作者更新入口

2026-11-08 起专用分支为 `collab/3b-4b`。仓库仍是 EthanY666999/decision-point。
本次只隔离交付范围，保留 11 月 3 日科学方案；新的实验设计尚未发布，不自动增加实验。
当前运行继续使用原代码完成，已完成项不重跑。本轮整理不代表 3B/4B GPU 验收已通过。

## 新目录获取（推荐）

```bash
git clone --depth 1 --single-branch --branch collab/3b-4b https://github.com/EthanY666999/decision-point.git ~/decision-point-3b4b
cd ~/decision-point-3b4b
git branch --show-current
git rev-parse HEAD
python3 code/scripts/verify_release.py
```

该分支只交付 3B/4B 所需代码、FrozenLake 示范、固定任务/评估依赖与文档。
`code/configs/frozenlake_05b.json` 是配置生成器使用的公共模板依赖，保留它不代表要运行 0.5B。
`results/supplemental500/` 的协议、题表和五个 PaTR 批次也是现有评估器的校验依赖，并非合作者的新结果。
模型权重、Python/CUDA 依赖须由内网已有目录或允许的镜像提供。

## 后续只更新这一条分支

以下命令只在**没有训练或评估使用此目录**且工作区干净时执行：

```bash
cd ~/decision-point-3b4b
git status --short
git pull --ff-only origin collab/3b-4b
python3 code/scripts/verify_release.py
```

有修改、分叉提交或 pull 拒绝时保留原目录，不执行 reset --hard、clean -fd、强制覆盖或盲目 stash pop。
现有 main 工作区不必切换：继续供旧运行使用；新版本克隆到上面的新目录。模型可复用绝对路径，无需再复制权重。
主仓库 main 以后只用于总导航和历史证据，不将其整支 merge 到本交付分支。

## 只读接收与离线使用

不要求 push、不要求 GitHub 写权限，不要求外传日志、轨迹或权重。
内部保存每次运行的实际配置、代码提交、模型版本、完整日志和评估记录；单位允许时才分享汇总数字。
可以通过批准渠道下载[本分支 ZIP](https://github.com/EthanY666999/decision-point/archive/refs/heads/collab/3b-4b.zip)，解压到新目录，运行同一校验命令。
ZIP 通过 TRACK.json、RELEASE_MANIFEST.json 识别交付；Git 用户另记录完整提交 SHA。

## 当前任务与下一版

- [保留的 3B/4B 任务书](PATR_3B_4B_REPRODUCTION_TASK_2026-11-01.md)：本次只更新获取分支，不改变方法、seed、预算和评估。
- 下一版任务书只在 `collab/3b-4b` 发布，并说明哪些已完成项可复用；不把 2B 的 Qwen3.5、FLH 或 Sokoban 设置自动套到 3B/4B。
- 不运行历史归档的调度器；本次 clone/pull 本身不会启动训练。
