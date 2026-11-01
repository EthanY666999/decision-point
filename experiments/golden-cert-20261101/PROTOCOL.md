# CERT预注册执行协议（2026-11-01）

用户明确采用TASK.md并要求优先腾盘、保持正在运行的B3。TASK.md保留原文；本协议记录可执行定义、审计和未验证项。

## 顺序与门槛
- B3 PaTR fresh400及其固定supp500 → CERT GPU一致性+G1 → 原B3 Random smoke/fresh400/supp500 → G1通过后才审查G2。
- 当前PaTR训练不停止、不换配置、不重启。仅将原CPU队列监督器交接给新CPU等待器；等待原PID和启动ticks对应的进程自然退出，取得同一supervisor.lock且GPU为空后再运行新任务。B3源码/配置不改。
- G1为三模型、3x64训练题、五选择器、F1/F2；两个模型SFT和SFTGRPO400用于gate，原SFTPaTR400只作参考。B1任务SeedSequence([42,batch+1,1701])和采样rng(20261031+batch)不变。
- G1 gate只用F2：UCT混合组严格多于Uniform，合并两模型按UID聚类bootstrap 20000次95%下界>0，不少于B1 K2两方法较大混合组数。token门槛保守要求同时不超过两个K2对照各自token的115%；不能用共享缓存摊薄每臂成本。
- 后续G2固定F2。F1仅描述，因任务书未给“足够”的可执行阈值，本次不据F1临时改预算。
- G2两臂独立原SFT fresh150，val25；门槛为100/125/150三点UCT严格高于Uniform且不低于B2PaTR；101–150混合组均值同时高于Uniform及B2精确值0.0721875，不能用四舍五入7.2%放宽。
- G2入口不读取/评估测试，保留冻结train.py，通过独立入口仅对final test调用记录not_evaluated_G2。G3/G4未准入，待上一层gate及新的时间容量核验；不承诺在本租期内做完全部seed。

## 实现与解释边界
- train.py、patr_core.py逐字节复制B2。新采样器复用其start_branch、apply_response、training_weights；不重分词模型输出、不重算前缀old_logprobs。环境新增观察文本仍按V1方式编码。V1指本地被验证的token管线，不是vLLM的V1引擎模式；冻结运行实际VLLM_USE_V1=0。
- R1，B0=4，每节点最多一次、每fork新增1叶；所有臂关闭score prune，repeat_limit6保留；初始概率按terminal success，剪枝和到达上限算失败。
- 以逐token前缀去重，canonical父为最小(j,turn)。新lineage=j@turn.0，与选择器及F无关；同节点suffix跨臂共用缓存，真实联合采样成本另列。
- first-token分布为给定已生成格式前缀的条件近似，不是精确动作边际分布。top20之外及非动作质量归invalid；格式缺失单计；首token动作但最终解析失败单计。无效动作经源码与CPU运行核验：原地、不奖励、不终止，因此归STAY。
- logprobs=0/1/20在同64题同种子必须逐位ids/mask/old_logprobs/actions/reward/terminal相同。0vs20满足任务书；额外1vs20对齐现有V1。2026-11-02首个SFT GPU验收未通过，已按此规则停止CERT。0vs20与1vs20均有实际token/action序列差异，不能放宽门槛；原始数据与差异审计已保留。
- U=(1-p_hat)S+p_hat Hcons是任务书指定的自适应混合，不具备经典UCT的访问次数/置信上界，名称不构成理论保证。
- CPU G0每回合最小采样token概率包含格式token，不等于Hcons/Hact；提供leaf-weighted主视图和prefix-unique附表，后者保留首次出现的结果条件标签，不用于推断或gate。回合/邻洞/墙边关联不证明因果。
- TASK E9的文字“都不高于K5”与所列71.75对68.0并不一致，且跨seed/步数/硬件有混杂；不沿用作结论。关闭score prune“几乎无影响”仍是待检验推测。相关工作表为用户提供待查证资料，不能直接当成创新性证据；BranPO已有失败回溯。
- G3任务书的末三点320/360/400与G2/B3 val25不同；G3尚未实现/启动，需在其阶段明确评估日程，不能悄悄替换。

## 磁盘、时限和证据
- 2026-11-01约19:33刷新Galaxy dashboard仍显示到期11-02 15:09:44，北京时间；中央deadline 12:09:44不变。未来G1和Random400启动前等待器会写rental_check_request.json并进入waiting_platform_refresh；20分钟巡检必须刷新平台并更新platform_verification.json（checked_unix晚于请求），才能放行。未验证续租不得延长。
- G1保守预留2h，外加后续Random400 6h、smoke/supp/余量45m。当前训练完成后若不足完整预算，停止启动新任务并报告capacity_pending，不缩预算、不抢占。
- 输出实际/data/patr-storage/golden-cert-20261101/code，兼容/home/vipuser/golden-cert-20261101链接；G1不写权重，所有新GPU准入仍要求输出盘16GiB。
- 本次已完成40文件处理：22个远端冗余副本移除前均重新完整本地SHA与远端SHA校验、查无使用者；18个B2正式final_model文件迁到data并逐个SHA核验，原路径链接保留。原始轨迹、日志、配置、评估及冻结旧归档未删。receipt在output/golden-cert-20261101/disk_20261101；被移走权重完整存在本地weights-backup目录。
- 原本GitHub轻量记录和新manifest先提交8be4c4f，再执行；权重不在GitHub。保护30文件及先前38迁移链接复核。
- 继续每20分钟中文报告，包含正常推进、训练窗口、固定验证、成本、梯度/更新、GPU/两盘/备份、故障和下一步；CERT未来训练增加p_hat/分叉回合/重复首动作比例。
