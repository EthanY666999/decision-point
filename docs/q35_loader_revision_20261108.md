# 2026-11-08 授权修订：S05 后的 F03 加载验收

S04、S05 必须在原 e32ac58 工作树串行完成全部最终评估。新工作树仅 CPU 准备。交接器检查两份400步完整 verification、hardware 源码提交、旧调度进程退出及 GPU/调度锁；仅归档带本次交接 token 的临时阻断记录及明确授权释放的 F03/H01-H05 原待人工记录，不终止运行进程。

顺序为已完成/运行的 S01–S05 → F03 额外评估 → H01 基座评估 → H02 SFT评估 → H03 GRPO4 → H04 Random K2 → H05 UCF-v2，之后保留原顺序。F03 禁止重训 SFT；单独 journal 最多一次兼容评估，原两次 attempts 不修改，不计入实验恢复额度。

## 加载检查

在统一 train.build_models / q35_engine_loader / sync_weights 返回后，F03评估、F03起点RL冒烟及正式训练均运行相同检查。独立 HF bf16 F03 reference 与实际 actor 的嵌入和末层权重一致；vLLM使用已同步actor的模型。固定10个原版 validation prompt及token IDs（artifacts/f03_loading_prompts.json），每个比较贪心生成的前最多8个token（EOS可提前结束）。HF对同一token序列teacher-force逐token计算原始logprob，vLLM取相同token的原始logprob。

预注册工程阈值：逐token最大绝对差≤0.125 nat，且每prompt平均绝对差≤0.02 nat；任何非有限值、缺token、模型配置或prompt哈希不一致均失败。阈值是bf16兼容验收容差，不代表已经通过GPU检查，禁止依据测试分数或失败结果自动调宽。完整HF/vLLM值、token、差值保存在loading_logprob_check.json。检查保存/恢复PyTorch、CUDA、Python、NumPy RNG及模型模式，推理请求单独显式seed，清空prefix cache。GPU实测只能在S05验收之后，当前CPU测试不能替代它。

## thinking 门禁

所有新启动检查prompt模板enable_thinking=False，初始与后续轮次均带空think块。对真实冒烟mask逐段复核，生成侧单独计数。原始 /data/models/Qwen3.5-2B 的生成标记只记录；F03-SFT或未知模型生成标记仍不通过、保留待人工。FLH三个基座冒烟的来源、prompt与真实输出已CPU复核；Random22个结束标记原始证据和旧失败门禁均保留。

H03/H04/H05各有实际F03起点的一步GPU冒烟（既有smoke参数），同样执行10prompt加载检查、有限指标与生成侧门禁；不复用基座冒烟作为SFT验收。所有训练算法、学习率、采样预算和正式评估配置与原计划逐项一致。固定验证不等于最终测试，test不用于选参。GPU故障恢复端到端仍是未完成事项；F37/F38和S11–S17不因本修订变为已完成。
