# CERT 独立探测版执行协议 v2（2026-11-02，GPU结果产生前固定）

## 版本与优先级
用户11月2日明确授权执行新附件。TASK.md保留新附件原文，本协议修正其中矛盾并作为实际执行定义。替代旧协议中跨logprobs逐位一致验收、生产top20采样和旧插队顺序；其余G1/G2收益门槛保持。旧golden-cert-20261101源码、任务书、结果和失败结论不改写。旧失败没有证明异步/bf16根因，未查证的相关工作、E9推论及“不是代码问题”不作为事实。

## 排程与保全
当前B3 Random fresh400及固定supp500完整结束，B3队列B3_seed42_complete且GPU无子进程后，新CPU等待器取得原supervisor.lock。之后新版技术验收→G1三模型→G1 gate→G2两臂150及gate。失败保留证据并停止相应CERT阶段；不中断Random，不放宽门槛。G3/G4必须上一层通过后单独做实现审计和完整时空准入，不凭本文件直接启动。

新代码实际路径/data/patr-storage/golden-cert-20261102/code，home兼容链接。train.py、patr_core.py保持与B2逐字节一致。旧PaTR500禁止续训。正式恢复用--resume且先查活进程；不会因为已按收据删除的完成checkpoint重训。

平台03:50实测刷新确认租期2026-11-03 15:09:44北京时间，中央截止更新为11-03 12:09:44，保留3小时备份。新长阶段仍需晚于rental_check_request的新刷新证据；每次准入实际输出盘至少16GiB。验收+G1完整预算3小时（验收1h、G1 2h规划；三模型共享总上限3h），G2两臂各4h加15min余量；预算不足capacity_pending，不静默减少题数/重复数/步数。硬时限退出保留部分结果，绝不将部分结果判为通过。

## 生产采样与独立探测
初始B0=4、T1、H10、repeat_limit6、单轨迹token上限、种子协议保持。初始直接调用冻结core.collect，设置branch_interval=H+1以禁用树分叉，前响应只读观察器复制完整branch；原apply_response处理模型输出。续写复用原stream参数、frozen apply_response、精确前缀断言，logprobs固定1；不强制换动作。生产ids/mask/old_logprobs不重分词、不重算。初始/续写均logprobs1，探测logprobs20，此条覆盖TASK 2.1/3.1开头/3.3的top20和偏差表“已经验证”。

初始全部完成后逐token前缀去重；每节点两条独立max_tokens1探测，前缀原ids和前缀+指定代码围栏。tokenizer实际编码围栏且断言单token，不盲信73594。动作首token映射包括无空格/空格变体。独立seed命名空间、请求ID、无并行生产请求。探测不调用apply_response、不执行环境动作、不进入优化器轨迹；保留raw top20概率。

概率质量：q=P(指定围栏token|prefix)，u_a=P(动作首token属于a|prefix+围栏)，代理p_a=q*u_a，残差p_invalid=1-q*sum(u_a)。不是完整最终解析动作边际：其他格式、top20缺失、首token后解析失败都可能造成偏差；缺失token、返回质量、代理与实际解析的校准单列。熵按后果FAIL/GOAL/STAY/不同状态聚合，除以log5。U=(1-p_hat)S+p_hat Hcons并非经典UCT置信上界。

每个独立臂成本=全初始+全候选探测+该臂续写；不摊薄共享探测成本。单列probe输入/生成token、request_seconds、wall_seconds、环境动作0。总真实联合采样成本另报。frozen初始仅有wall耗时，request_seconds不包括该部分，不把重叠请求时长称walltime。前缀缓存维持冻结builder设置，不能声称探测免费或只prefill。

## 新技术验收预注册
每模型使用第一批B1同64训练题，6组预固定种子stable_seed('cert-v2-acceptance',20261102,repeat)。每组A/B为冻结collect同设置重复，C为独立暖身采样+探测后的只读观察采样；A/B/C同base_seed，偶数ABC、奇数CBA。暖身及探测成本全部记录，暖身不加入比较或训练。记录A/B和A/C的ids、old_logprobs一致率，但不再要求跨运行逐位相同。原分叉前缀逐位相同和有限梯度断言仍硬性保留。

指标为success、平均回合及success/hole/turn_limit/token_limit/context_limit/pruned_repeat终局比例。按UID聚类重复，C-(A+B)/2的90%bootstrap等效CI（10000次，seed20261102）必须严格落在预固定界限：success/每终局±5pp，回合±0.5。并且每指标平均差绝对值不得超过6组A/B成对差的最大绝对值。交集验收，任一失败/不确定均不通过，不因结果临时扩大界限。此为新诊断筛查，并不证明所有种子/模型下采样严格等价。

校准：每动作0..4在同一完整canonical节点集比较p_a与最终解析动作indicator；按UID聚类bootstrap10000次(seed20261103)，各动作99%CI必须严格在±0.10内，5动作Bonferroni家族95%。单列均值、区间与代理局限。误差可能来自截断/格式代理或实现，失败不得直接断言方法无效或归因浮点。全三模型技术验收通过才产生完整G1 gate；一个模型失败停止新版后续训练。

## 收益门槛不变
G1三模型原SFT/SFTGRPO400/原SFTPaTR400参考；各3x64训练题，同B1SeedSequence([42,b+1,1701])和采样rng(20261031+b)，五选择器、F1/F2、同初始和同节点续写。R1，关闭score prune，保留repeat6，短缺计数且不得伪造叶。只用前两模型F2：UCT混合组严格多于Uniform，UID聚类bootstrap20000次95%下界>0；不少于B1K2较强者；生成token（包含全部探测）不超过每个B1K2对照115%。F1和其余选择器只描述，不据结果改G2预算。

G2固定F2，UCT/Uniform各从原SFT独立fresh150、seed42、val25，冻结优化器和奖励；禁止测试选择，末次test调用只记not_evaluated_G2。100/125/150三点UCT严格高于Uniform且不低于B2PaTR；101–150混合组均值严格高于Uniform和B2精确0.0721875。未通过不得进入G3。G3后续评估日程320/360/400需独立协议解决与val25差异，不能静默替换。

## 记录
每20分钟中文汇报正常进度、最近最多20完成训练步success/invalid/mixed均值范围、固定验证及变化、无新测试明确说、真实tokens/actions/time、梯度/参数更新、GPU/双盘/备份健康。CERT增加p_hat、分叉回合、dup_first_action_rate、独立probe成本。轻量同步既有GitHub decision-point分支；权重和优化器另做完整SHA备份，GitHub不含权重。新清理必须先备份和远端SHA核验且无使用者，活动checkpoint不复制不删除。
