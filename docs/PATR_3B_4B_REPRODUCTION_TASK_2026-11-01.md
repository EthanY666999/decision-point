# PaTR 3B / 4B 复现任务书

2026年11月1日 · @Ethan

## 一页看懂

目标：在 FrozenLake 上用 Qwen2.5-3B-Instruct 和 Qwen3-4B-Instruct-2507 复现 PaTR，每个模型做「无 SFT」和「SFT」两版；另外按论文在 SWE-Bench 上试 Qwen3-4B。全部沿用我们 0.5B 的代码和超参数，只换模型，其余一律不改。

硬件：1–2 张 80G 卡。两张卡时，两个实验各占一张卡并行跑，不做多卡训练。

### 实验清单（每个模型都跑一遍，seed 固定 42）

| 优先级 | 编号 | 起点 | 方法 | 每题轨迹 | 用途 |
| --- | --- | --- | --- | --- | --- |
| P0 | R1 | 原始模型 | GRPO8 | 8 | 对齐论文表 1 的 GRPO |
| P0 | R2 | 原始模型 | PaTR（K=5） | 约 4–6 | 对齐论文表 1 的 PaTR |
| P0 | R3 | 原始模型 | Tree-Random（K=5） | 约 4–6 | 对齐论文表 1 的 Random |
| P1 | S0 | 原始模型 | SFT（1000 条教师轨迹） | — | 得到 SFT 模型 |
| P1 | S1 | SFT 模型 | GRPO4 | 4 | 算力匹配基线 |
| P1 | S2 | SFT 模型 | PaTR（K=5） | 约 4–6 | SFT 版 PaTR |
| P1 | S3 | SFT 模型 | Tree-Random（K=5） | 约 4–6 | SFT 版 Random |
| P2 | R4 | 原始模型 | GRPO4 | 4 | 无 SFT 的算力匹配基线 |
| P2 | S4 | SFT 模型 | GRPO8 | 8 | 和 0.5B 的 SFT 结果对齐 |
| P3 | — | 同上 | 补 seed 43、44 | — | 时间够才做 |

### 做的顺序

1. Qwen2.5-3B：P0 → P1 → P2。论文有 3B 的数字，最有对比价值。

2. Qwen3-4B（FrozenLake）：P0 → P1 → P2。论文没有这个设置，是我们自己的规模点。

3. Qwen3-4B（SWE-Bench）：单独一条线，先只做评估，见第 5 步。

### 时间粗估

参考：0.5B 在 A100 上，SFT→PaTR 400 步约 3.5 小时，SFT→GRPO8 400 步约 6 小时。3B、4B 预计慢 3–6 倍，即每个树方法约 10–20 小时，GRPO8 约 20–35 小时。这是猜的：每个实验先跑 10 步，用实测速度乘 40，写进汇报表。

每个模型 P0+P1 共 7 个实验；两张卡并行时，一个模型大约需要 3–5 天。

## 第 1 步：环境和代码准备（约 1 小时）

全部在 Linux x86_64、Python 3.12 上做。必须用 codex/golden-sft-patr-20261031 分支，main 分支没有 SFT 代码。

1. 拉代码（两张卡就拉两份，目录名不同，原因见第 2 步）：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch codex/golden-sft-patr-20261031 \
  https://github.com/EthanY666999/decision-point.git ~/patr-ws0
cd ~/patr-ws0/code
```

2. 装环境（版本已锁死：torch 2.7.0、vLLM 0.9.2、transformers 4.51.3，都支持 Qwen3）：

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install bitsandbytes   # 第 2 步的显存降级可能用到
python -m pip check
```

3. 下载模型，并记下打印出来的 commit 号（写进汇报表）：

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download, model_info
for repo, d in [('Qwen/Qwen2.5-3B-Instruct', 'artifacts/Qwen2.5-3B-Instruct'),
                ('Qwen/Qwen3-4B-Instruct-2507', 'artifacts/Qwen3-4B-Instruct-2507')]:
    sha = model_info(repo).sha
    snapshot_download(repo, revision=sha, local_dir=d)
    print(repo, sha)
EOF
```

4. 校验数据和代码（每个模型各跑一次单元测试，全部通过才能继续）：

```bash
python ../golden-trace/verify.py
python scripts/golden_supplemental.py --model x --output x --audit-only
SFT_TEST_MODEL=artifacts/Qwen2.5-3B-Instruct     python -m unittest discover -s tests -v
SFT_TEST_MODEL=artifacts/Qwen3-4B-Instruct-2507  python -m unittest discover -s tests -v
```

Qwen3 的对话模板和 Qwen2.5 不同。如果 Qwen3 的测试报「前缀不一致」（assert 失败），停下来把报错发给我们，不要自己改模板：这会影响训练用的 token 是否和模型实际输出一致，我们之前就在这里崩过一整轮实验。

## 第 2 步：显存方案（每个模型定一次，之后不再改）

结论：原配置在一张 80G 卡上跑不下 3B 和 4B。训练器把训练模型（FP32）、参考模型和 vLLM 推理引擎放在同一张卡上，必须按下表降级。

### 显存估算（GB，不含激活）

| 部分 | 3B（Qwen2.5，3.1B 参数） | 4B（Qwen3，4.0B 参数） |
| --- | --- | --- |
| 训练模型 FP32 权重 | 12.4 | 16.1 |
| 梯度 FP32 | 12.4 | 16.1 |
| AdamW 状态（原版 / 8-bit） | 24.7 / 6.2 | 32.2 / 8.0 |
| 参考模型 BF16 | 6.2 | 8.0 |
| vLLM（0.30 / 0.18 / 0.22 × 80G） | 24 / 14.4 / — | 24 / — / 17.6 |
| 原配置合计 | 79.7（会 OOM） | 96.4（会 OOM） |
| 推荐配置合计 | 70.1（档位 1） | 65.8（档位 2） |

### 固定的降级顺序

| 档位 | 改什么 | 是否改变算法 |
| --- | --- | --- |
| 0 | 原配置 | — |
| 1 | vllm_memory_fraction 0.30 → 0.18；micro_batch_size 16 → 8；micro_batch_token_limit 24000 → 12000 | 不改，只是切得更碎、采样更慢 |
| 2 | 档位 1 + 优化器换成 8-bit AdamW（"optimizer": "adamw8bit"）；4B 的 vLLM 用 0.22 | 轻微改变，必须写进偏差表 |

- 3B 从档位 1 开始，OOM 再升到档位 2。

- 4B 直接用档位 2。

- 一个模型定下档位后，这个模型的所有方法、无 SFT 和 SFT 两版都用同一档，否则方法之间没法比。

### 档位 2 需要的代码改动（只改这一处）

在 code/ 目录下运行一次，打印 patched 即成功：

```bash
python - <<'EOF'
from pathlib import Path
p = Path('train.py'); s = p.read_text()
old = "    optimizer=torch.optim.AdamW(actor.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.)\n"
new = ("    if cfg.get('optimizer','adamw')=='adamw8bit':\n"
       "        import bitsandbytes as bnb\n"
       "        optimizer=bnb.optim.AdamW8bit(actor.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.)\n"
       "    else:\n" + old.replace('    optimizer', '        optimizer', 1))
assert s.count(old) == 1
p.write_text(s.replace(old, new)); print('patched')
EOF
```

改完后运行 sha256sum train.py，把结果写进汇报表。

### 怎么确定档位

不单独做显存测试。第一个正式实验（R2 PaTR）的前 10 步就是测试：

1. 用当前档位启动。

2. 若前 10 步内 OOM：删掉 runs/formal/patr_seed42，升一档，重新启动。

3. 若前 10 步正常：记下 train_metrics.jsonl 里的 peak_allocated_gib，以后不再改配置。

### 两张卡怎么用

代码一个工作目录只允许跑一个训练（有锁文件）。所以拉两份代码（~/patr-ws0、~/patr-ws1），各自装好环境，分别用 CUDA_VISIBLE_DEVICES=0 和 CUDA_VISIBLE_DEVICES=1 启动，各跑一个实验。不要让两个训练共用一张卡。

## 第 3 步：FrozenLake 无 SFT（R1–R4）

下面以 3B 为例。4B 把所有 3b 换成 4b、模型换成 Qwen3-4B-Instruct-2507、档位换成 2 即可。所有命令都在 code/ 目录下、激活 .venv 后运行。

### 3.1 生成配置文件（只做一次）

把第 2 步定下的档位填进 TIER：

```bash
python - <<'EOF'
import json
base = json.load(open('configs/frozenlake_05b.json'))
TIER = {'3b': 1, '4b': 2}          # 显存档位，按第 2 步的结果改
MODEL = {'3b': 'artifacts/Qwen2.5-3B-Instruct', '4b': 'artifacts/Qwen3-4B-Instruct-2507'}
for size in ['3b', '4b']:
    for start, model in [('nosft', MODEL[size]), ('sft', f'runs/sft_{size}/final_model')]:
        for group in [8, 4]:
            c = dict(base, model=model, flat_group_size=group, run_label=f'{size}_{start}_g{group}')
            if TIER[size] >= 1:
                c.update(vllm_memory_fraction=0.18, micro_batch_size=8, micro_batch_token_limit=12000)
            if TIER[size] >= 2:
                c.update(optimizer='adamw8bit', vllm_memory_fraction=0.22 if size == '4b' else 0.18)
            json.dump(c, open(f'configs/fl_{size}_{start}_g{group}.json', 'w'), indent=2)
print('done')
EOF
```

会得到 8 个配置，例如 configs/fl_3b_nosft_g8.json。g8 给 GRPO8、PaTR、Random 用（树方法不看这个值），g4 只给 GRPO4 用。

### 3.2 让评估脚本认分片权重（只做一次）

3B、4B 的权重会存成多个文件，而评估脚本只认单文件：

```bash
sed -i "s/(model \/ 'model.safetensors').is_file()/(model \/ 'config.json').is_file()/" scripts/golden_supplemental.py
sed -i "s/(model\/'model.safetensors').exists()/(model\/'config.json').exists()/" scripts/evaluate.py
grep -n "config.json" scripts/golden_supplemental.py scripts/evaluate.py   # 两个文件各出现一行才算成功
```

### 3.3 跑实验

每个实验都是同一套流程：启动 → 等它自己跑完 400 步和主测试 → 跑补充 500 题 → 改名归档。训练会自动每 40 步跑一次验证、结束时自动跑主测试 100 题。

| 编号 | --method | --config | 归档名 |
| --- | --- | --- | --- |
| R2 | patr | configs/fl_3b_nosft_g8.json | 3b_nosft_patr |
| R1 | grpo | configs/fl_3b_nosft_g8.json | 3b_nosft_grpo8 |
| R3 | tree_random | configs/fl_3b_nosft_g8.json | 3b_nosft_random |
| R4 | grpo | configs/fl_3b_nosft_g4.json | 3b_nosft_grpo4 |

先跑 R2（它的前 10 步顺便确定显存档位）。以 R2 为例：

```bash
mkdir -p logs runs/done
# 1) 启动（第二张卡在另一个工作目录用 CUDA_VISIBLE_DEVICES=1）
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py --method patr \
  --config configs/fl_3b_nosft_g8.json > logs/3b_nosft_patr.log 2>&1 &

# 2) 看进度
tail -n 2 runs/formal/patr_seed42/train_metrics.jsonl
cat runs/formal/patr_seed42/status.json

# 3) 跑完后（status.json 显示 complete），跑补充 500 题
CUDA_VISIBLE_DEVICES=0 python scripts/golden_supplemental.py \
  --model runs/formal/patr_seed42/final_model --output runs/formal/patr_seed42/supplemental500

# 4) 归档：删掉巨大的断点文件，把整个目录改名
rm runs/formal/patr_seed42/checkpoint.pt
mv runs/formal/patr_seed42 runs/done/3b_nosft_patr
```

- 一定要归档改名：代码的输出目录只按「方法_seed」命名，GRPO8 和 GRPO4、无 SFT 和 SFT 会抢同一个目录，不改名下一个实验会拒绝启动。

- 中途断了（停电、进程被杀）：用同一条启动命令加 --resume，从最近的断点接着跑。不要删目录重来。

- 硬盘：一个 3B 实验运行中约占 60–80 GB（断点含优化器状态），归档后约 20–30 GB。启动前确认至少有 100 GB 空闲。

## 第 4 步：FrozenLake SFT 版（S0–S4）

SFT 在强化学习之前单独做一次：用 1000 条教师轨迹（每回合只写一个动作）训练 1 个 epoch，学习率 2e-5，每批 16 条，共 63 步。超参数和 0.5B 完全一样，不要调。之后所有 SFT 版实验都从这个 SFT 模型出发。

### 4.1 做 SFT（S0，3B 约十几分钟）

```bash
# 1) 先审核：只检查数据和训练位置，不训练
CUDA_VISIBLE_DEVICES=0 python sft.py --model artifacts/Qwen2.5-3B-Instruct \
  --data ../golden-trace/sft_messages.jsonl --output runs/sft_3b_audit --audit-only

# 2) 正式 SFT
CUDA_VISIBLE_DEVICES=0 python -u sft.py --model artifacts/Qwen2.5-3B-Instruct \
  --data ../golden-trace/sft_messages.jsonl --output runs/sft_3b > logs/sft_3b.log 2>&1

# 3) 确认完成：应显示 complete，且 runs/sft_3b/final_model 里有 config.json
cat runs/sft_3b/status.json

# 4) 评估 SFT 模型本身（不做 RL）：主测试 100 题 + 补充 500 题
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --model runs/sft_3b/final_model \
  --output runs/sft_3b/main100.json
CUDA_VISIBLE_DEVICES=0 python scripts/golden_supplemental.py --model runs/sft_3b/final_model \
  --output runs/sft_3b/supplemental500
```

- 审核如果报 Unexpected teacher response 或 assert 失败（Qwen3 最可能），停下来把报错发给我们。

- 4B 做 SFT 如果 OOM，用下面这行把 SFT 的优化器换成 8-bit（只在 OOM 时改，并写进偏差表）：

```bash
sed -i "s/    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)/    import bitsandbytes as bnb; optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-5, weight_decay=0.0)/" sft.py
grep -n "AdamW8bit" sft.py   # 出现一行才算成功
```

### 4.2 SFT 之后的强化学习

流程和第 3 步完全一样（启动 → 跑完 → 补充 500 题 → 归档改名），只是换成 sft 配置：

| 编号 | --method | --config | 归档名 | 优先级 |
| --- | --- | --- | --- | --- |
| S2 | patr | configs/fl_3b_sft_g8.json | 3b_sft_patr | P1 |
| S3 | tree_random | configs/fl_3b_sft_g8.json | 3b_sft_random | P1 |
| S1 | grpo | configs/fl_3b_sft_g4.json | 3b_sft_grpo4 | P1 |
| S4 | grpo | configs/fl_3b_sft_g8.json | 3b_sft_grpo8 | P2 |

两张卡时的推荐排法：卡 0 跑 R2 → R3 → S2 → S3，卡 1 跑 R1 → S0（SFT）→ S1 → R4 → S4。

## 第 5 步：SWE-Bench（Qwen3-4B）

结论：先只做评估和可行性测试，不要直接开始完整训练。我们的代码库里没有 SWE-Bench 的代码；论文用的是一台 H200 服务器，1–2 张 80G 卡很可能跑不动完整的 300 步训练。这部分不做 SFT（没有现成的教师轨迹）。

### 论文的 SWE-Bench 设置（照抄，不改）

| 项目 | 论文设置 |
| --- | --- |
| 策略模型 | Qwen3-4B-Instruct-2507 |
| 训练数据 | R2E-Gym-Lite-with-Difficulty，去掉最难的题 |
| 评估 | SWE-Bench Verified，500 题，SWE-agent 脚手架，Docker 执行 |
| 最多回合 / 回复长度 | 50 / 32K token |
| batch / 步数 / 分叉间隔 | 8 / 300 / K=13 |
| 打分器 | Skywork-o1-Open-PRM-Qwen-2.5-7B（PaTR-PRM）或 Qwen2.5-Coder-7B-Instruct 当裁判（PaTR-Judge） |
| 硬件 | 一台 H200 服务器 |

### 论文的 SWE-Bench Verified 结果（解决率 %）

| 方法 | 解决率 | 环境正常结束 | 平均回合 |
| --- | --- | --- | --- |
| GRPO | 22.2 | 79.2 | 25.8 |
| DAPO | 24.6 | 71.4 | 31.6 |
| ARPO | 25.4 | 74.0 | 27.3 |
| Tree-Random | 24.8 | 84.4 | 30.1 |
| PaTR-Judge | 26.0 | 90.8 | 24.1 |
| PaTR-PRM | 27.2 | 92.2 | 24.4 |

### 分阶段做，每一阶段不达标就停

| 阶段 | 做什么 | 停止条件 |
| --- | --- | --- |
| W0 环境 | 按 SWE-bench 官方文档装 Docker 和评测工具；用官方自带的标准答案跑 5 题，全部判为通过 | 判不出通过 → 停，是环境问题 |
| W1 基线评估 | 用 vLLM 启动 Qwen3-4B-Instruct-2507，SWE-agent 跑 SWE-Bench Verified 全部 500 题（先跑 50 题看速度），记录解决率、平均回合、耗时 | 没有 — 这个数字论文没报，对我们有用 |
| W2 可行性 | 用 rLLM 的 SWE 训练示例跑 GRPO 3–5 步（batch 8），记录每步耗时和显存 | OOM，或一步超过 2 小时（300 步就要 600 小时以上）→ 停，把数字发回来 |
| W3 完整训练 | 只有 W2 通过、且拿到老师那边的 8 卡集群时才做：先 GRPO 300 步，再在 rLLM 里加 PaTR 树采样 | 与我们商量后再开始 |

- W0 和 W1 不占训练时间，可以和 FrozenLake 实验错开跑：只用一张卡做推理。

- Docker 镜像很占硬盘（几百 GB 量级），开始前按官方文档确认空间。

- 论文 SWE 部分的初始分支数、top-k 等细节，开始 W3 前以论文第 5 节原文为准。

## 第 6 步：要交回什么

每跑完一个实验，填下面表格一行，并把归档目录里的小文件发回来（权重不用发，但要保留）。

### 发回的文件

- config.json、status.json、train_metrics.jsonl、branch_events.jsonl

- validation_*.json、final_test.json / .jsonl（主测试 100 题）

- supplemental500/ 整个目录

- signal_stats.csv（下面的脚本生成）

- 训练日志 logs/*.log 的最后 200 行

- rollouts_*.jsonl.gz 不用发，但不要删，以后可能要重算

### 信号统计脚本

把下面存成 code/scripts/signal_stats.py，每个实验归档后运行 python scripts/signal_stats.py runs/done/`<归档名>`。它统计每一步里「全对、全错、有对有错」的题组比例，是我们现在最关心的指标。

```python
"""用法: python scripts/signal_stats.py runs/done/3b_nosft_patr  -> 写出 signal_stats.csv"""
import csv, gzip, re, sys
from collections import defaultdict
from pathlib import Path
run = Path(sys.argv[1]); rows = []
uid_re = re.compile(r'"uid": "([^"]+)"'); rew_re = re.compile(r'"reward": ([-0-9.eE]+)')
for f in sorted(run.glob('rollouts_*.jsonl.gz')):
    step = int(f.name[9:13]); g = defaultdict(list)
    for line in gzip.open(f, 'rt'):
        g[uid_re.search(line).group(1)].append(float(rew_re.search(line).group(1)) > 0)
    n = len(g); a1 = sum(all(v) for v in g.values()); a0 = sum(not any(v) for v in g.values())
    rows.append(dict(step=step, tasks=n, traj_per_task=round(sum(map(len, g.values())) / n, 2),
                     all_success=round(a1 / n, 4), all_fail=round(a0 / n, 4), mixed=round(1 - (a1 + a0) / n, 4)))
with open(run / 'signal_stats.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), 'steps; last:', rows[-1])
```

这个脚本已在我们 0.5B 的训练记录上核对过，和训练日志里记录的数字一致。

### 汇报表（每个实验一行）

| 归档名 | 模型 commit | 显存档位 | 显卡 | 主测试 Success / Pass@4 | 补充 500 Success / Pass@4 | 第 400 步验证 | 峰值显存 GiB | 总耗时 h | 第 201–400 步平均有信号题组 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3b_nosft_patr |  |  |  |  |  |  |  |  |  |  |
| 3b_nosft_grpo8 |  |  |  |  |  |  |  |  |  |  |
| 3b_nosft_random |  |  |  |  |  |  |  |  |  |  |
| 3b_sft（只 SFT） |  |  |  | — |  | — | — |  | — |  |
| 3b_sft_grpo4 |  |  |  |  |  |  |  |  |  |  |
| 3b_sft_patr |  |  |  |  |  |  |  |  |  |  |
| 3b_sft_random |  |  |  |  |  |  |  |  |  |  |

4B 照样再列一份。对照：论文 3B 的主测试是 GRPO 66.8 / 71.0、Tree-Random 72.7 / 75.0、PaTR 74.3 / 79.0。

## 第 7 步：和 0.5B 同步的消融与对比实验

目的：3B、4B 要和 0.5B 做同一套对比，这样才能看出「模型变大后结论变不变」。下表列出 0.5B 上做过的全部消融和对比，分成三类：你现在就能跑的、等我们给脚本再跑的、不用你跑的。

### 7.1 全部对应关系

| 编号 | 0.5B 上的实验 | 0.5B 的结果（补充 500 题或验证集） | 3B/4B 怎么做 | 谁准备 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| E1 | 主对比：GRPO8 / GRPO4 / PaTR / Random，无 SFT 与 SFT | 见第 3、4 步 | 已在第 3、4 步 | 你 | P0–P1 |
| E2 | SFT 后 PaTR 和 Random 补 seed43 | PaTR 79.2，Random 83.35 | 改 seed 重跑，见 7.2 | 你 | P1 |
| E3 | B2 / B3：K=2、每题叶子上限 8，SFT 起点 | B2 150 步验证 PaTR 71.75，Random 68.0；B3 进行中 | PaTR 和 Random 各 400 步 | 我们给脚本 | P1 |
| E4 | B1：只采样不训练，K ∈ {1,2,3,5} | 改 K 多 19 组有信号题组，换选择器只多 3–6 组 | 在 SFT 模型和 SFT→GRPO 模型上各采一次 | 我们给脚本 | P2 |
| E5 | 论文同款地图（map8，每回合步数上限 8）评估 | SFT→Random 84.25，SFT→PaTR 74.0 | 只评估，不训练 | 我们给脚本 | P2 |
| E6 | CERT（我们的新方法） | 0.5B 的 G1 还没出结果 | 0.5B 通过 G2 后再做 | 我们给脚本 | P3 |
| E7 | DAPO、ARPO（论文基线） | 0.5B 还没验收 | 0.5B 验收后再做 | 我们给脚本 | P3 |
| E8 | 按难度分层、剔除重叠的 409 题、按题配对置信区间、信号统计、跨显卡重评 | 见总结文档第 3 章 | 你只要发回第 6 步的文件 | 我们离线算 | — |

### 7.2 现在就能跑：seed43（E2）

在 P1 做完后跑，每个模型 2 个实验。先生成配置：

```bash
python - <<'EOF'
import json
for size in ['3b', '4b']:
    c = json.load(open(f'configs/fl_{size}_sft_g8.json'))
    c.update(seed=43, run_label=f'{size}_sft_g8_seed43')
    json.dump(c, open(f'configs/fl_{size}_sft_g8_s43.json', 'w'), indent=2)
print('done')
EOF
```

| 编号 | --method | --config | 输出目录 | 归档名 |
| --- | --- | --- | --- | --- |
| E2a | patr | configs/fl_3b_sft_g8_s43.json | runs/formal/patr_seed43 | 3b_sft_patr_s43 |
| E2b | tree_random | configs/fl_3b_sft_g8_s43.json | runs/formal/tree_random_seed43 | 3b_sft_random_s43 |

其余流程（补充 500 题、删断点、改名、信号统计）和第 3 步完全一样。注意输出目录名里是 seed43。

### 7.3 等我们给脚本：E3–E7

这几组实验在 0.5B 上用的脚本写死了我们服务器的路径和队列系统，不能直接拿来用。我们会整理成和第 3 步一样的「一条启动命令 + 一个配置文件」，放进仓库后更新这一节。你不用自己改。

先告诉你将来会跑什么，方便安排卡和硬盘：

| 编号 | 和主实验相比只改了什么 | 每个模型几个实验 | 单个耗时（估计） |
| --- | --- | --- | --- |
| E3 | 分叉间隔 K 5 → 2；每题叶子上限 8；每 25 步验证；推理后端 xformers | 2（PaTR、Random） | 比 K=5 多约 40% |
| E4 | 不训练，只用 3 批 × 64 道训练题采样统计 | 1 次，内含 4 个 K × 2 种选择方式 × 2 个模型 | 几小时 |
| E5 | 只评估，地图步数上限改成 8 | 每个已完成的 SFT 模型各 1 次 | 不到 1 小时 |
| E6 | 采样改成 CERT（先采 4 条再回溯分叉） | 2–4 | 待定 |
| E7 | 方法换成 DAPO 或 ARPO | 2 | 和 GRPO8 相近 |

做 E4 和 E5 时，用到的是 SFT 模型和 SFT→GRPO 模型的 final_model，这两个目录一定不要删。

## 规则和遇到问题怎么办

### 五条规则

1. 只换模型和显存档位。学习率 1e-6、KL 0.001、clip 0.28、温度 1.0、400 步、batch 64、树参数（B0=4、K=5、top-k=2、M=2）都不改。想改任何东西先问我们。

2. 训练中途不改配置、不提前停止、不挑最好的检查点。一律报第 400 步的结果。

3. 测试集只用来最后报告，不用来做任何选择。

4. 一张卡同时只跑一个训练。

5. 对代码的任何改动，都写进下面的偏差表，附上改动后的 sha256sum。

### 偏差表（边做边填）

| 改动 | 用在哪些实验 | 原因 | 改动后文件 sha256 |
| --- | --- | --- | --- |
| 评估脚本认分片权重（3.2） | 全部 | 3B/4B 权重是多文件 |  |
| 显存档位 1 | — | — |  |
| 显存档位 2（8-bit AdamW） | — | — |  |

### 常见问题

| 现象 | 怎么办 |
| --- | --- |
| 前 10 步 OOM | 删掉这个实验的输出目录，升一个显存档位重跑（第 2 步） |
| 第 10 步以后 OOM | 不要改配置；用 --resume 接着跑。连续 2 次都在同一位置 OOM 就停下来发日志 |
| Refusing to overwrite | 上一个实验没归档改名，先做 3.3 的第 4) 步 |
| Another training process holds the workspace lock | 这个工作目录已经有训练在跑，另一张卡要用另一个目录 |
| CUDA illegal memory access / Xid 31 | 我们 0.5B 也遇到过：换 xformers 后端、关掉 CUDA graph 后解决。发日志给我们，我们给具体改法 |
| 验证成功率持续下降、同一题 4 次输出完全一样、梯度范数上百 | 可能是训练崩了。不要停，继续跑完，但马上告诉我们 |
| Qwen3 单元测试或 SFT 审核报 assert | 停。不要改模板或跳过检查，发报错给我们 |
| 硬盘快满 | 先删已归档实验的 checkpoint.pt；不要删 final_model、rollouts_*、评估结果 |

### 联系节点

以下三个时间点主动发一次汇报表：第一个实验跑完前 10 步（确定显存档位）、每个模型的 P0 三个实验跑完、每个模型的 P1 跑完。
