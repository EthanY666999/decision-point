# Qwen2.5-0.5B FrozenLake：GRPO / PaTR / Tree-Random

这是实验用的 **PaTR论文方法独立重实现**，不是论文作者官方完整训练代码。原始复现实验保留2026-10-16/17完成训练的代码，本更新另附2026-10-31 Golden Trace预热实验；旧版本的可执行代码已从当前目录移除，保留精简报告/指标与Git历史。

## 从哪里开始

- **运行代码：[`code/`](code/)，完整步骤见[复现指南](docs/REPRODUCE.md)。** 基础模型可按固定commit下载；数据和所需rLLM环境代码已随仓库提供。
- **结果：[`results/`](results/README.md)**，区分原100题主结果与新增500题补充评估，包含训练指标、验证/测试原始轨迹、统计分析和来源证据。
- **方法与限制：[`docs/METHOD_AND_LIMITATIONS.md`](docs/METHOD_AND_LIMITATIONS.md)**。
- **历史问题：[`history/`](history/README.md)**，只供追溯，不作为运行入口。
- **打包验证：[`docs/VALIDATION.md`](docs/VALIDATION.md)**；文件校验清单见`RELEASE_MANIFEST.json`。

## 快速开始

在Linux x86_64服务器、Python 3.12、NVIDIA A100 40GB或兼容GPU上执行。需要匹配PyTorch CUDA运行时的驱动、C/C++编译器和Python开发头文件。Mac CPU/MPS不能执行本项目的vLLM训练。

```bash
git clone --depth 1 https://github.com/EthanY666999/decision-point.git
cd decision-point/code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python scripts/prepare_model.py
python -m unittest discover -s tests -v
python scripts/run.py --method grpo --smoke
```

冒烟检查只用验证集的8题完成初始化、1次训练更新及输出检查，不读取正式测试集。通过后执行正式基线：

```bash
python scripts/run.py --method grpo
python scripts/run.py --method patr
python scripts/run.py --method tree_random
```

逐个执行，每种方法均从同一基础模型独立训练400步。PaTR不从GRPO最终权重继续训练。不要在同一GPU同时启动多份。先验收GRPO基线，再比较树方法；运行命令不会自动认定任何基线已通过科学验收。

## 已完成结果（seed42）

| 方法 | 原100题 Success | 原100题 Pass@4 | 补充500题 Success | 补充500题 Pass@4 |
|---|---:|---:|---:|---:|
| V1 GRPO（沿用既有模型） | 61.00% | 62.00% | 63.60% | 64.20% |
| 当前PaTR | 64.50% | 71.00% | 73.00% | 77.80% |
| 当前Tree-Random | 64.75% | 68.00% | 67.10% | 69.80% |

两套测试独立报告，不合并，不用补充集替换主结果。500题并非全部独特/未见过的地图几何。仅一个训练seed；主100题不足以证明PaTR优于Random，补充500题的配对区间支持该抽样集上的优势，不代表跨训练seed稳健性。

所有方法都存在首回合无效动作，长度也与论文存在明显差异，因此不宣称完整复现论文指标。环境保留无效动作并继续交互，结果未修复格式或重新评分。

## 文件与权重

仓库不含GB级基础模型、训练后权重、优化器checkpoint及800个完整训练rollout。基础模型由下载脚本获取；训练后权重由上述命令重新生成。若只想评估已训练模型，需要另向项目所有者获取`final_model/`目录。现有评测轨迹和训练指标可以直接阅读，无需GPU。

代码整理没有修改`train.py`、`patr_core.py`、冻结配置、任务清单及环境源码；它们的SHA256登记在`code/SOURCE_MANIFEST.json`。新加的是便携启动/模型准备/评估入口和发布验收。历史租期截止文件和服务器控制器没有作为默认运行配置发布。

## Golden Trace SFT 实验已完成

[Golden Trace数据](golden-trace/README.md)已接入独立SFT入口`code/sft.py`。SFT之后的GRPO400和PaTR400从同一个SFT权重独立开始；PaTR再续训至500步并停止。原始基线及其训练入口保留。

**[查看完整结果与原始评估轨迹](results/golden-sft-20261031/README.md)**：SFT→PaTR400主100题73.50%/75.00%、补充500题75.85%/77.60%；PaTR500分别79.75%/81.00%和80.00%/81.00%（Success/Pass@4）。单seed且训练预算不同，不宣称等算力或论文超越。

**[查看新增审计与对照计划](experiments/golden-controls-20261031/PLAN.md)**：补充500剔除91道训练几何重叠题后单独报告409题；3+1分支分配的离线证据不足，未启动训练。状态探针与SFT→Tree-Random400、GRPO组大小4对照正在执行，尚无最终对照结果。

完整操作历史见[EXPERIMENT.md](EXPERIMENT.md)。`code/SOURCE_MANIFEST.json`与`RELEASE_MANIFEST.json`仍标识原发布版本；新增结果和脚本由`GOLDEN_SYNC_MANIFEST.json`单独校验。
