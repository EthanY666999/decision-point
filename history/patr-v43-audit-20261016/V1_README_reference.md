# PaTR FrozenLake 0.5B 复现

**模型：Qwen2.5-0.5B-Instruct；设备：单张 A100-SXM4-40GB。**

这是独立的论文方法重实现。原 rLLM FrozenLake 环境与 Agent 保留；作者未公开或未确认的配置明确列在 [复现审计](REPRODUCTION_AUDIT.md)。不把烟测当完整训练，不把原文数字当我们的实测。

远程目录：`/home/vipuser/patr-reproduction`。

本地代码目录：`/Users/ethan/Documents/ChatGPT/ACL2027/outputs/patr-reproduction`。

## 当前执行范围

GRPO、PaTR、Tree-Random、DAPO、ARPO，各从原始模型开始，seed=42，正式训练各 400 步、每步 64 题。FrozenLake 最多 10 turns、累计响应上限 10240 tokens、K=5。先做真实参数更新烟测，通过后启动后台正式队列。所有方法的最终评估均为 100 题 × 4 次采样。

运行结果和状态以 [RESULTS.md](RESULTS.md)、`runs/formal/*/status.json` 和 `train_metrics.jsonl` 为准；烟测结果单列在 `runs/smoke/`。只有 `complete` 才表示该方法完成 400 步和最终测试。

## 文件

- `patr_core.py`：数据、原始环境/Agent 接口、五类 rollout、组内 advantage 与评估统计。
- `train.py`：vLLM 采样、单卡全参数 GRPO 更新、检查点、独立验证和最终测试。
- `configs/frozenlake_05b.json`：冻结方法/训练配置。
- `vendor/`：引用上游环境及对照算法源码、许可证。
- `tests/`：有针对性的算法/数据测试。
- `artifacts/`：模型、数据、依赖与审计 manifest。
- `runs/`：方法独立的训练日志、轨迹、检查点、最终权重和指标。
- `logs/`：依赖安装、下载、执行器和各方法原始日志。
- `scripts/verify_final.py`：从保存的轨迹重新统计评估指标、重放环境动作，并核验每方法 400 步与最终 100 题 × 4 条独立轨迹。

## 使用

```bash
cd /home/vipuser/patr-reproduction
.venv/bin/python -m pytest -q tests
.venv/bin/python train.py --method grpo --smoke
.venv/bin/python train.py --method grpo
# 意外退出后恢复同一方法，不会重置初始模型
.venv/bin/python train.py --method grpo --resume
```

`--smoke` 只做小批次 1 步、1024 token 上限，用来验证链路，不产生论文结论；开发早期也保留了 10240 token 的烟测记录。正式模式才是配置中的 400 步。

第 1 步完成时保存检查点，之后每 10 步保存；SIGTERM/SIGINT 会在本轮迭代结束后安全保存。请不要用 `kill -9`。`artifacts/deadline.json` 可设 unix 时间，让后续迭代自动停止并保存；开始最终评估前也检查窗口，若已过期则保留第 400 步断点，待恢复后完整执行 100 × 4 评估。无需保持 Cursor/SSH 窗口开启。

本轮数据下载直接连接官方 Hugging Face，已经验证；训练/评估加载本地权重，不需要持续联网或外部 API。

## 首个正式批次实测

2026-10-13，GRPO 的第 1/400 步完成：64 题、512 条轨迹，采样 269.4 秒，整步 362.0 秒，生成 318827 tokens，训练组成功率 7.23%，峰值 PyTorch 分配 24.46 GiB。此前 100 题 × 4 条的未训练验证成功率为 5.75%。这些是训练起点与吞吐校准，**不是最终复现指标**。

若后续维持首步速度，单方法 400 步约 40 小时（尚未计入全部评估开销），五方法全套会超过一天。随着生成长度变化，速度也会变化；不会为赶租期而把未完成步骤写成完整实验。每 10 步保存检查点，继续运行受 `artifacts/deadline.json` 限制。正式队列已从步骤 1 的检查点继续。

## 论文参照数值

以下来自 [PaTR 表 1](https://arxiv.org/html/2607.15610v1#S5.T1)，**不是本轮结果**。

| Method | Success % | Pass@4 % | RLen | Turns |
|---|---:|---:|---:|---:|
| GRPO | 66.5 | 67.0 | 513 | 3.4 |
| DAPO | 71.0 | 71.0 | 562 | 4.2 |
| ARPO | 69.7 | 72.0 | 377 | 3.3 |
| Tree-Random | 56.3 | 63.0 | 400 | 3.9 |
| PaTR | 75.8 | 78.0 | 344 | 3.6 |

复现质量需要同时考察配置差异和实测趋势；单种子不能建立显著性结论。

## 查看后台任务

```bash
cd /home/vipuser/patr-reproduction
bash scripts/status.sh
cat artifacts/queue_status.json
tail -f logs/formal_grpo.log
```

正式队列由 `scripts/run_queue.py` 顺序运行，各方法初始权重独立；某方法报错或暂停时队列停止，先诊断恢复该方法。只有完成 400 步、最终评估并通过原始轨迹核验后，才运行下一方法。不要在已有训练进程运行时重复启动队列。

```bash
# 只在没有现存队列时启动/恢复
cd /home/vipuser/patr-reproduction
tmux new-session -d -s patr-formal 'env -u CUDA_LAUNCH_BLOCKING .venv/bin/python -u scripts/run_queue.py >> logs/queue.log 2>&1'
```

队列启动时设一个 24 小时断点保护窗口，用于限制无人值守运行；这不是已确认的云实例到期时间。窗口到达会保存并暂停，不会关闭云实例或自动续费。若五种方法未全部完成，报告会明确显示剩余步骤。单张卡的全部正式训练是否在一天内完成，以正式 batch 的实际耗时为准。

自动跟进：本聊天每 20 分钟检查一次，仅在重要进展、故障、完成或需要用户处理时通知。服务器训练不依赖本地窗口；自动诊断需要本地电脑及 ChatGPT 应用保持运行。SSH 已配置本机专用密钥，密码不写入仓库或脚本。

2026-10-13 已由“实验2”聊天接续跟进。用户确认额外续租一天后，保存暂停保护窗口延长一次至北京时间 **2026-10-15 05:23:22**；最新值以 `artifacts/deadline.json` 为准。这是保存暂停保护时间，不是自动查询到的云平台到期时间。变更记录见 `logs/deadline_updates.jsonl`。

正式训练全部结束后的核验命令：

```bash
.venv/bin/python scripts/verify_final.py --require-complete
```

该命令只审计已有记录，不生成测试样本或调整参数。训练期间可用 `--include-validation` 核验已有验证集记录；未完成的方法始终显示 pending。输出为 `reports/evaluation_audit.json`。

本地增量同步可执行 `bash scripts/sync_from_a100.sh`。脚本只从 A100 下载代码、报告、日志和轨迹，排除模型权重、训练检查点与依赖环境，使用 SSH 连接复用和有限次数重试。2026-10-13 两次新建同步连接关闭，同期服务器日志确认大量失败登录连接和 `MaxStartups` 限流；复用已认证连接后同步恢复，训练未中断。未修改服务器的登录或防火墙策略。

2026-10-13 06:48（北京时间），GRPO 完成第 37 步后在第 38 步采样发生 CUDA illegal memory access，内核记录 Xid 31。第 30 步检查点通过完整 CRC 检验；从该步恢复，重算 31–37 步。旧队列误启动的 PaTR 在初始验证后保存第 0 步安全退出。恢复时通过 `vllm_enforce_eager=true` 关闭 CUDA Graph，并使用 `CUDA_LAUNCH_BLOCKING=1` 定位可能再次发生的内核错误；这不是已证明 CUDA Graph 为根因。此次工程执行方式变化和重算轨迹可能产生数值差异，算法、样本预算、奖励、精度与最终评估协议不变。已回滚的记录保存在对应 run 的 `recovery_history/`，故障与恢复记录见 `logs/engineering_recovery.jsonl`。

08:40，第 92 步采样再次发生 Xid 31；同步堆栈定位到 `_vllm_fa2_C.fwd_kvcache`，因此关闭 CUDA Graph 本身不足以解决问题。队列正确停止，未跳过 GRPO。保留失败证据并从第 90 步断点恢复；工程覆盖增加 `vllm_attention_backend=XFORMERS`，使用已安装的 xformers 0.0.30 / vLLM 0.9.2 PagedAttention 解码路径，避开报错的 FA2 KV-cache 内核。保持 bf16 推理、相同采样参数和完整训练/评估预算。此后端变更可能影响浮点数值与采样结果，最终对照需披露 GRPO 的分阶段执行后端；根因是否完全消除仍待后续运行验证。

XFormers 首次使用 Triton 前缀内核需要 Python 开发头文件。该镜像缺少 `Python.h`，已将与系统 Python 完全匹配的 Ubuntu `libpython3.12-dev=3.12.3-1ubuntu0.17` 解包到 `artifacts/build-deps/`；训练进程通过 `C_INCLUDE_PATH` 使用这些头文件，不执行系统安装或改变 Python/Torch 版本。此构建依赖不随报告同步到本地，来源和校验值另存审计记录。

15:09，在 XFormers 连续运行至第 325 步且无新增 Xid 后，安全保存第 325 步断点，并通过完整 CRC 和 650 次优化器更新核验。恢复时取消临时的 `CUDA_LAUNCH_BLOCKING=1` 同步调试开关，以去除排障期间的额外同步开销；其用途见 [PyTorch 2.7 文档](https://docs.pytorch.org/docs/2.7/cuda_environment_variables.html)。仍使用 XFormers 和 eager 模式，训练源码哈希与此前相同，没有丢弃已完成迭代或改变冻结协议。实际运行表现和恢复检查见 `reports/cuda_debug_mode_audit.json`。

GRPO 约第 300 步后出现较多 `Goal` 无效移动指令和少量长输出。前 307 步共 157184 条训练轨迹已逐条重放，题目序列、动作、奖励和统计一致；第 320 步完整验证的无效动作率为 30.73%。这是需要在结果中披露的策略行为，未通过修改解析器、奖励或超参数处理。证据见 `reports/grpo_action_format_audit.json` 和 `reports/grpo_training_trajectory_audit.json`。

2026-10-14 01:43（北京时间），GRPO 与 PaTR 均已完成 400 步、800 次 AdamW 更新及最终 100 题 × 4 次测试，并通过 `scripts/verify_final.py` 核验。GRPO 最终 Success / Pass@4 为 61.00% / 62.00%，PaTR 为 60.00% / 61.00%；GRPO 全部 204800 条训练轨迹也已重放通过。PaTR 全程记录 400 个连续迭代，更新均有限且非零；这项统计核验不同于全量训练轨迹重放。其训练轨迹第 252、320 步已做环境重放与 token 解码抽查，第 357 步另做完整批次 token/文本及解析器核对。证据见 `reports/patr_training_completion_audit.json` 和相应 spotcheck 文件。其余方法仍需继续，不能据此宣称五方法复现完成。

PaTR 后期出现大量代码块结束符不完整的动作输出：最终测试 1882 次动作中 801 次无效（42.56%），全部为 Up / Down / Right / Left 的代码块结尾只有两个反引号。第 400 步验证为 68.50% / 69.00%，这是独立 validation 指标，不能替代最终 test 的 60.00% / 61.00%。原始解析器、奖励、训练步数和采样协议保持不变，没有据此调参或选择中途断点。最终测试格式审计见 `reports/patr_final_test_format_audit.json`；队列已在核验 PaTR 后推进至 Tree-Random，实时状态以 run 文件为准。

01:54，Tree-Random 第 2 步因 `Token id 151725 is out of vocabulary` 退出，队列正确停止。该 ID 小于模型词表大小 151936，但大于 tokenizer 的最大 ID 151664；vLLM 0.9.2 的旧输入校验拒绝了模型自己生成的词表位置。依据 [vLLM 官方输入校验实现](https://docs.vllm.ai/en/latest/api/vllm/renderers/base/)，通过 `scripts/apply_vllm_vocab_compat.py` 对固定依赖做最小兼容回移：校验同时考虑模型词表，并拒绝负数和真实越界。没有删除、替换或屏蔽生成 token，没有修改 tokenizer、logits、采样、奖励或算法。33 项回归检查和单条合成输入的 GPU 回送验证通过，证据见 `reports/vllm_vocab_compat_audit.json`、`reports/vllm_vocab_gpu_probe.json`。原依赖源文件及哈希保留在 `artifacts/engineering/vllm_vocab_compat/`。

此次失败前只完成 1 步，旧保存策略尚未产生检查点，因此归档原尝试后从原始权重重算；废弃的第 1 步为 329.37 秒、200916 个生成 token，另外还有初始验证和未完成第 2 步成本，见 `runs/formal/tree_random_seed42/recovery_history/20261014_vocab_reinjection_step0002/incident.json`。02:00 同一队列重新启动 Tree-Random。训练入口增加首步完成即保存，降低下一次早期故障的损失；这不改变参数更新或实验预算。恢复是否通过故障点以实时日志为准。

## 从干净 Ubuntu 环境安装

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/apply_vllm_vocab_compat.py
python3 scripts/download_model.py
.venv/bin/python scripts/audit_maps.py
.venv/bin/python -m pytest -q tests/test_core.py
.venv/bin/python -m pytest -q tests/test_gpu.py
```

本次服务器完整依赖版本见 `artifacts/requirements.lock.txt`。GPU/驱动需支持所安装 PyTorch 的 CUDA runtime；当前使用 PyTorch 2.7.0+cu126、vLLM 0.9.2、Transformers 4.51.3。无需修改服务器系统 CUDA 或 base Conda。

2026-10-14 16:47（北京时间）核验：Tree-Random 已完成 400 步、800 次 AdamW 更新、最终 100×4 测试及环境重放，400 步断点完整 CRC 通过。最终 Success / Pass@4 为 43% / 45%，无效动作率 21.1752%；400 步 validation 为 51.5% / 54%，不能替代 test。队列已自动推进 DAPO，首步完成并在进行第 2 步；首步含 4 批拒绝重采样、2048 条实际采样轨迹，耗时约 1280 秒，额外采样成本按日志记录，不缩短协议。ARPO 尚待执行。最新 deadline.json 保存暂停窗口为北京时间 2026-10-16 05:23:22，本次跟进未延长窗口。
