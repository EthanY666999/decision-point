# PaTR 3B / 4B 精简复现任务书

2026-11-03 更新 · @Ethan。替代本文件 11 月 1 日的任务安排。**只用 main；只做 FrozenLake；一张 A100 80GB 串行。**

## 先看变化

本轮要尽快回答两个问题：3B/4B 上 PaTR 与随机树、独立采样如何比较；Golden Trace SFT 之后这种排序是否变化。0.5B 的部分对照显示 SFT 后收益明显，也出现 Random 强于 PaTR 的情况，所以保留完整对照，不预设 PaTR 一定更好。

- 已完成的实验与评估直接登记，不因更新重跑；当前正常运行的实验先完成，下一项再按新清单安排。不采用“超过一半必须完成、不到一半直接删除”的规则。
- **取消本轮新启动**：无 SFT 的 GRPO4、SWE 全线、K=2/B1 扫描、map8、CERT、DAPO、ARPO。已经产生的结果保留并标记为旧计划。
- **暂不排队**：SFT→GRPO8、额外 seed43/44。主线完成后再讨论，不自动启动。
- 合作者不用手改 train.py、sft.py 或评估脚本；main 已包含所需支持。
- 内网只需 clone/pull。完整证据本地保留，不要求把任何内部文件传出来。

## 1. 要做哪些实验

两个模型：`Qwen/Qwen2.5-3B-Instruct`、`Qwen/Qwen3-4B-Instruct-2507`。两者本轮都只跑 FrozenLake。3B 对应论文表 1 的模型；4B FrozenLake 是我们的扩展，不是原论文的 SWE 结果。

| 编号 | 起点 | 方法 | 每个模型是否必做 | 说明 |
| --- | --- | --- | --- | --- |
| R2 | 原始模型 | PaTR，K=5 | 是 | 过程评分选分支 |
| R3 | 原始模型 | Tree-Random，K=5 | 是 | 同树框架的随机选择对照；仍披露剪枝差异 |
| R1 | 原始模型 | GRPO8 | 是 | 每题独立采 8 条 |
| S0 | 原始模型 | 1,000 条 Golden Trace，SFT 一轮 | 是 | 每个模型各做一次，再评估 SFT-only |
| S2 | 对应 S0 | PaTR，K=5 | 是 | 从同一 SFT checkpoint 独立训练 |
| S3 | 对应 S0 | Tree-Random，K=5 | 是 | 同上 |
| S1 | 对应 S0 | GRPO4 | 是 | 组大小对照，不称为严格等 token/等算力 |

主线总量：**12 次 RL × 400 批 + 2 次 SFT**，另含固定评估；不是 14 次 RL。已完成项目从剩余工作量中扣除。删减主要来自取消 SWE 和附加实验，不通过缩短主实验、少做对照或按成绩提前停来省时间。

串行顺序（完成项跳过）：

1. 3B：R2 → R3 → R1 → S0 → S2 → S3 → S1。
2. 4B：S0 → S2 → S3 → S1 → R2 → R3 → R1。
3. 两条主线完成即停止本轮队列，汇总结果；额外实验不自动继续。

不承诺几天内完成。记录前 10 批含验证的耗时和后续稳定步耗时，估算剩余训练、验证与测试时间；初始化、验证和保存耗时不能简单忽略。

## 2. 获取代码与准备环境

只克隆一次 main。已有工作区、旧分支或 ZIP 用户先看[更新说明](COLLABORATOR_UPDATE.md)。正在训练时不要更新代码。

```bash
git clone --depth 1 --single-branch --branch main https://github.com/EthanY666999/decision-point.git ~/patr-ws0
cd ~/patr-ws0/code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
mkdir -p logs runs/done
```

需要 Linux x86_64、匹配 CUDA 的驱动、C/C++ 编译器及 Python 开发头文件。核心依赖沿用 torch 2.7.0、vLLM 0.9.2、transformers 4.51.3；这不是已验证 3B/4B 全流程的保证，下面的 tokenizer 审核和 GPU 验收仍要在目标机器做。

若使用档位 2，另安装内部批准版本的 `bitsandbytes`，记录确切版本，并再运行 `pip check`。内网不能访问 PyPI/Hugging Face 时，用内部镜像或批准的离线安装包，不要求向外上传信息。

模型可放在 `code/artifacts/Qwen2.5-3B-Instruct` 和 `code/artifacts/Qwen3-4B-Instruct-2507`。也可沿用内部已有的绝对路径，生成配置时用 `--model /内部模型目录`。记录模型 revision/来源；若 revision 不可查，记录文件哈希，不猜 commit。仓库本身不含权重。

若允许在线下载，只下载当前需要的模型并固定 revision：

```bash
python - <<'PY'
from huggingface_hub import model_info, snapshot_download
repo = 'Qwen/Qwen2.5-3B-Instruct'  # 做 4B 时改成 Qwen/Qwen3-4B-Instruct-2507
sha = model_info(repo).sha
snapshot_download(repo, revision=sha, local_dir='artifacts/' + repo.split('/')[-1])
print(repo, sha)
PY
```

先做 CPU 校验：

```bash
python scripts/verify_release.py
python ../golden-trace/verify.py
python scripts/golden_supplemental.py --model unused --output unused --audit-only
python -m unittest discover -s tests -v
```

未设 SFT_TEST_MODEL 时，通用 CPU 测试会跳过两项模型模板检查；这不是模型模板验收通过。每个模型必须单独运行：

```bash
SFT_TEST_MODEL=artifacts/Qwen2.5-3B-Instruct python -m unittest discover -s tests -p test_sft.py -v
# 到 4B 时运行对应检查，不能跳过
SFT_TEST_MODEL=artifacts/Qwen3-4B-Instruct-2507 python -m unittest discover -s tests -p test_sft.py -v
```

绝对模型路径同样可用。若模板前缀或目标 token 检查失败，保留错误并暂停该模型；不删断言、不重分词凑长度、不照搬其他模型模板。

## 3. 显存档位与配置：不用手改源码

原训练器同时持有 FP32 actor、梯度、优化器、BF16 reference 和 vLLM。3B/4B 的原始配置可能接近或超过 80GB，不能仅凭参数量认定一定能跑。

| 档位 | RL 优化器 | vLLM 显存比例 | micro batch / token 上限 |
| --- | --- | --- | --- |
| 1 | 原 AdamW | 0.18 | 8 / 12000 |
| 2 | 8-bit AdamW | 3B：0.18；4B：0.22 | 8 / 12000 |

3B 先用档位 1，4B 先用档位 2。保持 XFORMERS + eager。8-bit 优化器改变数值实现，必须记录；即使不改目标函数，microbatch 切分也不能承诺逐位一致。同一模型的六个 RL 对照采用同一档位；已有实验与新档位不一致时单列差异，不冒充完全匹配。

生成独立本地配置，不修改 Git 跟踪文件：

```bash
python scripts/prepare_collaborator.py --size 3b --tier 1 --output configs/local/3b
python scripts/prepare_collaborator.py --size 4b --tier 2 --output configs/local/4b
# 已有内部模型时在对应命令追加 --model /绝对路径/模型目录
```

每个目录生成 `nosft_g8.json`、`sft_g8.json`、`sft_g4.json`。g8 用于 GRPO8 和两种树方法；树方法的初始分支数仍是 4。脚本只生成配置，不会自动排队。相同内容可重复生成；不同配置拒绝覆盖，换档位请用新目录，例如 `configs/local/3b_tier2`。

先用当前模型第一个 PaTR 正式运行的前 10 批观察资源占用，之后继续同一次运行到 400。如果 OOM：保留日志和失败目录，先查显存及竞争进程；确认需要升档时，归档失败尝试，从原始起点用新配置重开，不能换优化器后 resume。10 批后仍可能 OOM，早期成功不是全程保证。

SFT 默认仍为 AdamW。4B SFT 如经检查需要 8-bit，使用 `sft.py --optimizer adamw8bit` 单独记录，失败尝试保留。SFT 优化器和 RL 优化器分别记录，不用 sed。

## 4. 运行 RL：一套命令覆盖所有主实验

所有命令在 `code/`、已激活 .venv 后执行。一张卡同时只运行一个 SFT/RL/评估进程。

以 3B 无 SFT PaTR（R2）为例：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/run.py --method patr \
  --config configs/local/3b/nosft_g8.json > logs/3b_nosft_patr.log 2>&1 &
```

查看进度：

```bash
tail -n 2 runs/formal/patr_seed42/train_metrics.jsonl
cat runs/formal/patr_seed42/status.json
```

每次正式运行 400 个任务批次，每批 64 题、两次优化器更新；每 40 批固定验证，完成后自动评估主测试 100 题×4 次。400 批不等于 400 次 optimizer.step。

| 编号 | `--method` | 3B 的 `--config` | 归档名 |
| --- | --- | --- | --- |
| R2 | patr | configs/local/3b/nosft_g8.json | 3b_nosft_patr |
| R3 | tree_random | configs/local/3b/nosft_g8.json | 3b_nosft_random |
| R1 | grpo | configs/local/3b/nosft_g8.json | 3b_nosft_grpo8 |
| S2 | patr | configs/local/3b/sft_g8.json | 3b_sft_patr |
| S3 | tree_random | configs/local/3b/sft_g8.json | 3b_sft_random |
| S1 | grpo | configs/local/3b/sft_g4.json | 3b_sft_grpo4 |

4B 把配置目录 `3b` 换为 `4b`，归档名前缀也改为 4b；如果另建显存档位目录，使用实际配置路径。S 系列必须先完成第 5 节 SFT。

status 为 complete、训练进程已退出且主测试完整后，运行补充 500 题。传入同一实验配置的工程参数，避免评估退回 0.5B 的显存设置：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/golden_supplemental.py \
  --model runs/formal/patr_seed42/final_model \
  --runtime-config configs/local/3b/nosft_g8.json \
  --output runs/formal/patr_seed42/supplemental500
```

`--runtime-config` 只读取显存、microbatch、执行后端等工程字段，不改变固定测试题、采样次数、评估种子和解码参数。脚本已支持带 index 的分片 safetensors，并检查每个分片存在；不再要求修改脚本。

补充集 status 为 complete、summary.json 有 500 题/2000 条轨迹后，本地统计并归档：

```bash
python scripts/signal_stats.py runs/formal/patr_seed42
# 确认目标目录尚不存在，且没有进程继续使用该运行目录
test ! -e runs/done/3b_nosft_patr && mv runs/formal/patr_seed42 runs/done/3b_nosft_patr
```

输出按方法和 seed 命名，所以每次完成必须归档，防止无 SFT/SFT、3B/4B 和 GRPO8/4 抢同一目录。不要覆盖、不要强行删除旧目录。**默认保留 checkpoint，不要求删除断点来交付。**磁盘不足时，先在内部验证备份和 final_model 完整，再决定是否清理已结束运行的可恢复断点；唯一证据不能删除。

中断时用相同代码、相同配置与相同命令追加 `--resume`。不要将不完整 final_model 当成完整 checkpoint。临时故障恢复与改算法重新实验分开记录。

磁盘按模型、优化器、轨迹实测；至少留出下一次原子 checkpoint 写入所需空间，不能只看最终权重大小。3B/4B 全参数运行可能需要数十至上百 GB，不把固定“100GB 足够”当保证。

## 5. SFT：每个模型一次，然后三个 RL 独立开始

使用仓库已有 **1,000 条 FrozenLake** 成功示范；不是 SWE 数据，不调用 GPT 重新生成。每回合仅监督动作和结束 token，1 epoch、batch16、microbatch1、学习率2e-5，共63次更新；不截断、不 packing。

以 3B 为例（内部模型路径可替换）：

```bash
python sft.py --model artifacts/Qwen2.5-3B-Instruct \
  --data ../golden-trace/sft_messages.jsonl --output runs/sft_3b_audit --audit-only

CUDA_VISIBLE_DEVICES=0 python -u sft.py --model artifacts/Qwen2.5-3B-Instruct \
  --data ../golden-trace/sft_messages.jsonl --output runs/sft_3b > logs/sft_3b.log 2>&1

cat runs/sft_3b/status.json
```

审核模式只读数据和 tokenizer，不训练、不创建输出目录。正式 SFT 若需 8-bit 则加 `--optimizer adamw8bit`；先确认 bitsandbytes 可用，不中途改优化器。正式 SFT 不提供断点恢复入口，失败时保留目录、从基础模型在新输出目录重跑；只有完整的 final_model 才能作为后续 RL 起点，并同步调整三个 SFT RL 配置的 model 路径。

SFT 完成后本身也评估：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --model runs/sft_3b/final_model \
  --runtime-config configs/local/3b/sft_g8.json --output runs/sft_3b/main100.json
CUDA_VISIBLE_DEVICES=0 python scripts/golden_supplemental.py --model runs/sft_3b/final_model \
  --runtime-config configs/local/3b/sft_g8.json --output runs/sft_3b/supplemental500
```

4B 用 `Qwen3-4B-Instruct-2507`、`runs/sft_4b`、`configs/local/4b`。完成 SFT-only 评估后再按顺序跑 S2、S3、S1。三种方法的 actor、reference 和 vLLM 均从同一个对应模型的 SFT checkpoint 开始，各用新 RL 优化器；绝不能 PaTR 接着 GRPO 终模训练。

原始模型的额外测试暂不新增。无 SFT 主线各运行里的 validation_0000 已提供固定验证起点；它不等于测试集成绩。需要基础模型 test 时另行安排，不把不同 split 混在一列。

## 6. 内网本地留存与简短汇报

**不用上传任何文件。**内部保存：实际 config、Git commit（ZIP 用户保存 manifest）、模型来源/文件哈希、pip freeze、训练日志、指标、分支事件、所有 validation、主100/补500原始评估、rollouts、source_snapshots、final_model 和 checkpoint。若单位允许分享，只需填写下表或提供允许外传的错误摘要；其余保留在内部。

| 实验名 | 状态/批次 | 代码与模型版本 | RL/SFT优化器、显存档位 | 主100 Success/Pass@4 | 补500 Success/Pass@4 | val400 Success | 峰值显存GiB | 训练耗时h | 生成token/动作数 | 201–400混合组% | 偏差/失败 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3b_nosft_patr（示例） |  |  |  |  |  |  |  |  |  |  |  |

每个实际实验一行；SFT-only 填两套测试，RL专属列填“不适用”。未完成的写真实批次，未评估写“未评估”，不能填0。记录总耗时的起止口径，step_seconds 与包含初始化/验证/保存的墙钟时间分开。

`signal_stats.py` 已随 main 提供，按完整 JSON 中的最终 reward 统计全对、全错、混合组比例；不再复制正则脚本。201–400 缺批次会标为不完整。混合组代表组内有奖励差异，不等于最终效果好，也不意味着无混合组时整个损失为0（KL仍可能贡献）。

主100与补500单独报告；补500含已知RL训练地图重叠。过滤409、按题配对置信区间等只有内部保存逐题结果后才能分析，本轮不要求外传原始数据，也不要求另做跨GPU推理。单训练seed不作跨seed稳健结论。

论文3B表1参考：GRPO 66.8/71.0、Tree-Random 72.7/75.0、PaTR 74.3/79.0（Success/Pass@4）。本地题集、短动作输出、额外SFT和8-bit优化器均可能不同，不能据数字接近就宣称完全复现。[原论文](https://arxiv.org/html/2607.15610v1)

## 7. 执行边界与遇到问题

- 固定：seed42、400批、batch64、mini batch32、学习率1e-6、KL0.001、clip0.28、温度1.0、10回合；树B0=4、K=5、top-k=2、M=2。不按测试成绩调参、挑checkpoint或改变任务顺序。
- 短期 reward 起伏不直接判失败；结合持续趋势、有限梯度、参数更新和原始动作。NaN、模板/mask不一致、参数未更新、持续计算故障则保留证据并暂停对应实验，不硬跑满400。
- OOM先检查资源与竞争进程；真正需要改档位时从原起点新开，记录旧失败。别反复 resume 同一个确定性 OOM。
- CUDA/Xid错误保留日志；不要临时修改多个设置后声称同一实验。更换后端也记入偏差。
- Qwen3模板失败不跳过；main合并、CPU校验通过不代表已替朋友完成GPU适配验收。
- 无法外发日志时，在内部留存错误原文，按规定描述错误类型与发生阶段；不要求绕过内部规定。

建议四个交流节点：每个模型第一个RL运行前10批完成；每次SFT及其评估完成；3B主线完成；4B主线完成。不能对外交流时改为内部记录。

SWE 由另外的工作线处理，本任务书不再提供可误启动的 SWE 命令。取消本轮某个消融是为了节省排期，不代表0.5B已经证明它在3B/4B上必然成立。研究协议与历史记录保留，但不增加本任务书之外的待办。
