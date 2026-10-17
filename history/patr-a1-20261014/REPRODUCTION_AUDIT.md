# PaTR / FrozenLake / Qwen2.5-0.5B-Instruct 复现审计

## 本轮能验证什么

这是**依据论文的独立重实现（paper-based reimplementation）**。截至本轮准备，未取得可确认的 PaTR 作者训练代码、配置与数据清单。因此可以验证公开算法在固定协议下的训练行为与指标；不能据此宣称逐行复现作者代码，或保证重现表中数值。单种子结果也不等于统计显著性结论。

不加入 Golden Trace、动态 K、新树算法或新奖励。这些未来研究改动与本轮隔离。全部模型动作由 Qwen 生成，BFS 不提供动作答案；只供 PaTR 训练期分配分支。评估全部使用独立完整轨迹，不调用过程评分器。

## 来源与冻结版本

- [PaTR 论文 v1](https://arxiv.org/html/2607.15610v1)，主要参考算法 1、公式 1/2/5/9、表 1 和附录 A。
- [rLLM 原始环境与 Agent](https://github.com/agentica-project/rllm/tree/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4)，commit `7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4`。FrozenLake 环境、提示词与动作解析源码原样放在 `vendor/rllm_source/`，保留其许可证。
- [ARPO 作者实现](https://github.com/RUC-NLPIR/ARPO/tree/e484f1ddc7b158acbc9c8cf5d03e8e2ccc6b4349)，commit `e484f1ddc7b158acbc9c8cf5d03e8e2ccc6b4349`，对照 entropy rollout 与 core algorithms 源码；相关源文件与许可证保留在 `vendor/arpo_source/`。
- [DAPO 官方实现说明](https://github.com/verl-project/verl-recipe/tree/main/dapo)。本轮实现 PaTR 文中指定的 token-level loss、动态过滤和非对称 clipping；不擅自增加未说明的长度奖励。
- [Qwen 官方模型](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，revision `7ae557604adf67be50417f59c2c2f167def9a775`，下载文件哈希见 `artifacts/model_manifest.json`。

## 算法与训练协议

五种方法分别从同一初始模型开始，seed=42，每种 400 次采样/训练迭代，batch=64 个任务，学习率 1e-6，最多 10 个环境交互、每条轨迹累计最多 10240 个 assistant token，训练温度 1.0。GRPO/DAPO 每题 8 条独立轨迹；树方法初始 4 条、展开为 2 个子分支，PaTR/Random 的间隔 K=5，margin=0.5，循环阈值 6。这些是本轮固定配置。

选择 K=5 是 FrozenLake 的设置；K=13 对应 SWE-agent。本轮不运行 SWE-Bench，也不下载 3B/4B 策略或 PRM。

对于题目 x 的完整、失败、早停轨迹组 G(x)，使用最终任务奖励：

\[
A_i=(r_i-\mathrm{mean}_{j\in G(x)}r_j)/(\mathrm{std}_{j\in G(x)}r_j+10^{-6}).
\]

标准差采用样本标准差（ddof=1）。所有 assistant token（包括共享前缀）参与训练；环境观测与提示 token 被 mask。零方差组的 policy-gradient advantage 为零，保留 KL 项。共享前缀按其在各叶轨迹中的出现参与 loss，与普通 trajectory GRPO 接口一致。

\[
L_{PG}=-\frac1{|X|}\sum_x\frac1{|G(x)|}\sum_i\frac1{N_i}
\sum_{t\in\mathrm{assistant}(i)}\min(\rho_{it}A_i,\mathrm{clip}(\rho_{it},1-\epsilon_l,1+\epsilon_h)A_i).
\]

\[
L=L_{PG}+0.001\,\mathbb E[\exp(\ell_{ref}-\ell_\theta)-(\ell_{ref}-\ell_\theta)-1].
\]

GRPO、PaTR、Random、ARPO 的 clipping 范围为 1±0.28；DAPO 为 [0.8,1.28]，用整个优化 minibatch 的有效 assistant token 数归一化，同时不断采样到获得 64 个 reward 有方差的组。过滤掉的采样成本计入总成本；32 个 batch 仍凑不够就明确报错，不退化成 GRPO。DAPO 未加入原始 DAPO 可选的 overlong reward shaping，因此只能称为 **PaTR 描述下的 DAPO 对照重实现**。

PaTR 在第 5 步对仍活跃分支评分，先选 top-2 替换为双子分支，再对未选中的分支做低分/重复动作剪枝，其余继续。结束、失败与剪掉的轨迹全部参与组内 advantage。因第 10 步到达上限，通常只有一次中间展开。

ARPO 移植作者代码中的统计量：前 20 个生成 token 位置上返回的 top-10（及 sampled token）logprob，累加 `-p log p / log(|V|)`；它不是完整词表 Shannon entropy。沿每条分支使用初次生成的统计量为基准；新子分支重设基准。`clip(U - 0.2*(H_current-H_initial),0,1) <= 0.5` 时新增子分支，总轨迹预算 8；未耗尽预算且路径已结束时补采根轨迹。FrozenLake 适配的这些参数并非 PaTR 作者公开配置。

## 未公开细节与本轮明确选择

| 项目 | 本轮选择 / 限制 |
|---|---|
| FrozenLake scorer 精确公式 | `clip(r_cumulative + clip(1-d_current/d_start,0,1),0,1)`；d 是 holes blocked 的 BFS 距离，不可达进度为 0；这是可审计重建，不是作者确认公式 |
| FrozenLake 展开 top-k | 2；原文未给出 FrozenLake 的确切值 |
| Tree-Random | 均匀随机选择展开分支，保留相同的过程低分剪枝条件；这是控制展开选择的对照，不能称为完全无过程信息的基线 |
| 数据/地图 | 使用上游数据生成 seed=42、10000 train/100 test 的顺序；尺寸 2–9，frozen tile probability 0.6–0.85，非滑动环境。这里的 p 不是滑动概率 |
| 地图 max_steps | 生成合法地图时设为 10；上游示例训练脚本有 8，论文写最大交互 10，作者 map-generator 参数不可确认 |
| 数据划分 | task UID 不交叉；小地图可能几何重复，数量见 `artifacts/map_audit.json`；此测试不是 geometry-held-out generalization |
| Validation | 另用 seed=20261013 生成 100 个任务；每 40 步评估；最终 test 只在 400 步完成后使用，不以 test 选 checkpoint |
| 评估解码 | 每题 4 次，temperature=0.7/top_p=0.8/top_k=20，来自上游评估习惯；PaTR 作者具体解码参数未确认 |
| 优化器/批次 | AdamW，betas=(.9,.999)，eps=1e-8，weight_decay=0，32 题 optimizer minibatch、单次采样更新 1 epoch、梯度裁剪 1；作者未完整公开 |
| 数值实现 | FP32 master weights/Adam 状态，bf16 autocast，bf16 固定 reference 与 vLLM 推理；无 LoRA、量化或蒸馏 |
| 框架 | 原样复用 rLLM env/agent，单卡 PyTorch+vLLM 重建训练编排；不同于作者未公开的 rLLM 分布式训练补丁 |
| 重复次数 | 第一轮 seed=42；未完成多个 seeds 前不报置信度提升或显著优势 |

## 验证与报告要求

1. CPU 检查：环境状态克隆隔离、分支替换与失败保留、随机 tie 行为、GRPO/DAPO 权重、指标定义、多轮 token/mask 一致性。
2. GPU 检查：生成真实动作、完整环境交互、有限 loss/gradient、非零参数变化、权重同步和前缀 cache 清空、保存与恢复 checkpoint。
3. 所有正式方法用相同数据、初始权重与评估协议。保存 configuration、模型/源码/数据哈希、依赖锁定、原始轨迹、分支事件和训练统计。
4. 训练阶段记录 success、tokens、turns、wall time、显存、loss、KL、clip fraction、gradient norm、参数变化。`sampled_surprisal` 不是 full policy entropy。
5. 最终报告 success rate（全部 400 条完成/截断轨迹的成功比例）、Pass@4（100 题中至少一次成功比例）、response length（assistant token 总数，含 EOS）、turns、失败/截断分类与实际开销。
6. `complete` 只用于 400 步与最终测试均完成的正式方法；烟测、暂停、失败、部分训练都不能冒充完整复现。结果可以高于或低于原文，不能为对齐表格反复调 test。

## 基础设施恢复规则

micro-batch、micro token 数、vLLM 并发和 vLLM 显存份额允许通过 `artifacts/runtime_override.json` 调整，以处理工程级显存问题。2026-10-13 GRPO 第 38 步出现 CUDA 非法内存访问（Xid 31）后，按用户已授权的工程故障恢复范围，增加 `vllm_enforce_eager` 执行模式开关，以关闭 CUDA Graph 排障；未确认它是故障根因。实际值写入每个 run 的 config/checkpoint。恢复时所有其他配置必须一致，否则拒绝恢复。执行模式变化不保证逐 bit 相同轨迹，需在最终报告披露。推理权重同步会逐次检查一个完整层权重与 actor 的 bf16 投影一致，并清空旧 prefix cache。

本轮单卡完整范围仅包括本文 FrozenLake 的 0.5B 五方法表格，不包括 SWE-Bench、3B 或论文中所有消融。未公开的细节仍需作者确认。

2026-10-13 08:40 的第二次非法访问在 `enforce_eager=true`、`CUDA_LAUNCH_BLOCKING=1` 下定位到 FA2 的 `fwd_kvcache`。工程恢复允许 `vllm_attention_backend` 在已支持的 `FLASH_ATTN` / `XFORMERS` 间选择；本次从第 90 步改用 XFormers 的 PagedAttention 解码路径，不升级依赖或改变模型、精度、算法及采样预算。GRPO 前 90 步与此后的执行后端不同，可能存在浮点数值/轨迹差异，必须作为对照限制披露。不能把切换后暂时通过若干步表述为已证明底层根因完全修复。

15:09，从完整核验的第 325 步断点恢复时，取消了 `CUDA_LAUNCH_BLOCKING=1` 的同步调试开关，保留 XFormers、eager 模式与相同训练源码。第 31–325 步使用此同步调试设置，第 326 步起恢复常规 CUDA 调度；此差异需纳入运行时间与工程恢复记录。所有完成的训练迭代均保留，没有检查点回滚。详见 `reports/cuda_debug_mode_audit.json`。

训练行为审计：GRPO 第 300 步后较多生成 `Goal` 等非法移动指令，并出现少量达到既定 token 上限的轨迹。前 307 步共 157184 条训练轨迹重放通过；第 320 步验证的 400 条轨迹也全部重放通过，成功率 65.25%、Pass@4 68%、无效动作率 30.73%。这些是中途行为与验证指标，不是最终测试。解析器仍是原始 rLLM 实现，没有把 `Goal` 接受为合法动作，也没有据此调整奖励或选取较高验证分的断点。最终报告应同时披露成功指标和输出格式问题。

完成阶段审计（2026-10-14）：GRPO 的全部 400 步、204800 条训练轨迹已重放通过，最终测试 Success 61.00%、Pass@4 62.00%、无效动作率 22.64%。PaTR 的 400 步连续性、全部有限且非零的更新、最终断点完整 CRC 和 800 次 AdamW 更新已核验；最终 100×4 测试重放通过，Success 60.00%、Pass@4 61.00%、无效动作率 42.56%。PaTR 最终测试 1882 次动作中的 801 次无效输出，全部缺少代码块闭合所需的第三个反引号。第 252、320 步训练轨迹抽查包含环境重放与 token 解码核对，第 357 步 289 条轨迹的 1320 个回答片段也全部与记录文本相符，所有无效回答片段自然结束于 EOS。树的共享前缀在叶轨迹中重复出现，按叶计数不可冒充调度器实际执行次数。证据分别保存在 `reports/patr_training_spotcheck_*.json`、`reports/patr_token_format_spotcheck_0357.json` 和 `reports/patr_final_test_format_audit.json`。

这轮单种子 PaTR 的最终 Success / Pass@4 均比 GRPO 低 1 个百分点，比论文 PaTR 参照值分别低 15.8 / 17.0 个百分点；不据此作统计显著性或算法普遍优劣结论。第 400 步 validation 的 68.50% / 69.00% 不能作为 test 成绩。格式问题已记录，未修改解析器、奖励、提示词或超参数，也未用最终测试选择断点。上述 PaTR 训练抽查和更新统计不等于对全部 125273 条训练轨迹完成环境重放；最终评估的 400 条轨迹则已全部重放。

2026-10-14 01:54，Tree-Random 的第 2 步触发 vLLM 0.9.2 输入词表校验错误：生成的 ID 151725 属于模型的 151936 个输出/embedding 位置，却超出 tokenizer 最大 ID 151664。此时无新增 Xid，训练及队列均已退出。对依赖 `_validate_model_input` 做官方后续实现的兼容回移，允许模型已有词表位置并保留负数、越界、空输入及长度检查；完整输入 token 不变，不屏蔽 logits，不更改采样分布、解析器或奖励。修复只作用于后续进程的请求校验；已完成的 GRPO/PaTR 没有重跑或修改结果。实际补丁、原依赖备份及 SHA、33 项回归检查结果和 GPU 合成输入验证分别见 `scripts/apply_vllm_vocab_compat.py`、`reports/vllm_vocab_compat_audit.json`、`reports/vllm_vocab_gpu_probe.json` 和故障归档。

Tree-Random 当时没有有效训练检查点，已归档并重算第 1 步，不把重复计算混入有效的 400 步；额外成本单列。训练入口仅额外增加首步完成时保存断点，原有每 10 步保存、每 40 步验证、最终 400 步及 100×4 协议不变。不能把此次 token 编号兼容故障与此前 PaTR 的代码块格式错误混为一谈：前者导致程序退出，后者是原始解析器下的实际策略输出结果。

02:15，恢复后完成 Tree-Random 第 2 步并继续第 3 步；第 1 步新断点完整 CRC 与 2 次优化器更新核验通过。初始验证的全部 400 条轨迹、第 1 步全部采样轨迹和分支事件均与旧尝试逐字节一致。然而第 1 步的 loss、KL、梯度等优化统计存在差异，原因尚未证实，不能宣称重算结果逐位一致；差异原值见 `reports/tree_random_step1_recovery_comparison.json`。恢复后的第 2 步未出现模型独有 token，故该完整迭代通过证明训练继续运行，但不能单独证明其再次执行了原失败 token 的回送路径；该路径已由独立 GPU 合成输入验证。详见 `reports/tree_random_vocab_recovery_verified.json`。

各方法的基础训练题目流由独立的 `(seed, step, 1701)` RNG 决定，optimizer minibatch 顺序由 `(seed, step, 1702)` 决定，避免树分支消耗的随机数改变下一批题目。DAPO 额外拒绝采样仍单独计费。工程开发阶段的烟测早于这一正式数据流固定，不能横向比较烟测分数。

按算法 1，仅在 K 的倍数且分支仍未终止时路由。因此本轮 K=5、max_turns=10 时，通常在第 5 步路由一次；当时尚不可能连续重复 6 次，Θ=6 的剪枝条件在这一 FrozenLake 配置中不会单独触发。若作者实际在每一环境步检查重复动作，则会与此算法级重建有差异；本轮不默默添加这种未确认实现。

## 异步推进的等价边界

正式运行使用 vLLM `LLMEngine.add_request/step`，某条轨迹的一轮生成结束后即可提交下一轮，不等待所有题目最慢的回答。PaTR/Random 仍保留每题 K-turn barrier；ARPO 仍保留每题每轮的分支分配 barrier。开始优化前等待整个训练 batch 完成，策略权重在整批采样期间固定，因此没有引入异步策略滞后。

每次 collect 固定一个 base seed；生成种子由 base seed、task UID、lineage、turn 的 SHA256 决定，路由使用每题独立 RNG 和稳定分支次序。不同题目的完成先后不会决定随机分支或采样种子。对五种方法分别用正序/逆序完成的模拟引擎测试，轨迹、分支和计数一致。

开发时的旧 `collect_batched` 保留为审计参考，正式路径为 `collect`。初次按轮次同步执行的 100 题初始验证耗时约 297 秒，尚未做任何正式参数更新就停止、归档；不会把它混入新调度的训练指标。

2026-10-14 16:47（北京时间）核验：Tree-Random 已完成 400 步、800 次 AdamW 更新、最终 100×4 测试及环境重放，400 步断点完整 CRC 通过。最终 Success / Pass@4 为 43% / 45%，无效动作率 21.1752%；400 步 validation 为 51.5% / 54%，不能替代 test。队列已自动推进 DAPO，首步完成并在进行第 2 步；首步含 4 批拒绝重采样、2048 条实际采样轨迹，耗时约 1280 秒，额外采样成本按日志记录，不缩短协议。ARPO 尚待执行。最新 deadline.json 保存暂停窗口为北京时间 2026-10-16 05:23:22，本次跟进未延长窗口。
