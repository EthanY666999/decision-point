# 内网使用与单分支更新

2026-11-03。所有当前复现入口均在 main；[精简任务书](PATR_3B_4B_REPRODUCTION_TASK_2026-11-01.md)是唯一任务清单。

## 新机器：一次克隆

```bash
git clone --depth 1 --single-branch --branch main https://github.com/EthanY666999/decision-point.git ~/patr-ws0
cd ~/patr-ws0/code
```

无需 GitHub 写权限，不需要 push。请从内部允许的渠道准备模型及 Linux/CUDA 对应依赖。训练和评估使用本地模型，不需要外部教师 API。

## 已有 main：一次更新

先让正在运行的训练/评估完成或正常保存退出。在仓库根目录：

```bash
git status --short
git pull --ff-only origin main
```

若有本地代码修改或 pull 拒绝，不使用 reset --hard、clean -fd 或强制覆盖。按下一节保留修改。训练期间不 pull，不改变正在执行的源码。

## 从旧实验分支迁移：在原克隆内完成

先记录 `git rev-parse HEAD`、`git status --short` 和当前分支，等待运行结束。新版本已有可选 8-bit AdamW 与分片权重支持，不必重新应用旧手工补丁。

若 tracked 文件有修改，先保存到本地 stash（不上传）；未跟踪的文件不在这个 stash 里，保持原位。如果还有无法判断用途的修改，先停在这里核对。

```bash
git stash push -m before-main-20261103
git fetch origin main:refs/remotes/origin/main
git switch main
git merge --ff-only origin/main
```

没有本地 main 时，`git switch main` 可从唯一的 `origin/main` 建立跟踪分支；若工具提示不能自动建立，使用 `git switch -c main --track origin/main`。若本地 main 有分叉提交，保留分支并核对，不强行重置。

`git stash list` / `git stash show -p` 可查看旧补丁；不要盲目 `stash pop` 把旧 sed 修改覆盖回来。原有 runs/、模型、日志和本地配置都要保留，不要求复制或重新训练。使用新版本前检查实际配置及输入模型路径。

**已有运行的恢复**：未完成的实验继续用它原来的代码与配置恢复；不要为了这次合并中途换优化器或模板。完成后再更新并开始下一实验。

## 内网不能上传：本地保存就是交付的一部分

- 不要求上传 checkpoint、权重、原始轨迹或日志；不要求把内网仓库 push 回 GitHub。
- 完整 config、代码版本、环境版本、训练/评估记录保留在内部目录。
- 可按单位规定分享汇总表中的数字、运行是否完成及错误摘要；不允许对外提供时只做内部留存。
- 409 题过滤、按题置信区间等需要逐题记录才能重算。拿不到原始记录时，应由内部运行相应分析或明确“未计算”，不能从一个成功率反推。
- 不把跨显卡重新评估写成离线统计；它是额外推理任务，本轮未安排。

只能下载压缩包的环境可使用 GitHub 的 main ZIP：<https://github.com/EthanY666999/decision-point/archive/refs/heads/main.zip>。通过内部批准的渠道传入后，解压到新目录，保留旧目录；不覆盖正在运行的工作区。ZIP 不带 Git 元数据，用 RELEASE_MANIFEST.json 与下面的校验命令识别版本。

```bash
python code/scripts/verify_release.py
```

仓库包含历史证据，完整 main 体积大于运行代码。无需下载旧分支或服务器大权重归档。历史任务书中残留的旧服务器路径、日期和分支描述不代表当前执行要求。
