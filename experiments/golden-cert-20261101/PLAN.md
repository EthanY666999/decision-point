# 2026-11-01 CERT接续与磁盘处理（核验至19:36）

用户采用CERT任务书。B3原PaTR训练PID146367自18:36:36持续运行，未发送训练停止信号。19:36:41完成55/400，较上次25新增30步；最近20步36–55成功率62.465%（52.793–76.133%）、无效动作0.21895%（0–0.62176%）、混合题组8.90625%（3.125–14.0625%）。第50步新固定验证68.25/70，较25步64.75/67提升3.5/3pp；无新测试。全部梯度有限且有参数更新，窗口梯度0.3685–2.35995，参数变化6.631e-7–1.156e-6。周期119943生成token、23430215输入token、29979动作、步耗时1405.92秒。GPU唯一146367，28687/49140MiB、60℃、瞬时100%。训练批次不作测试。

磁盘处理按manifest预先发布GitHub 8be4c4f后执行：40文件，22个已完整本地SHA验证的远端冗余副本移走（B2两正式checkpoint、Random B2 smoke全权重、PaTR B3 smoke全权重），18个B2正式final_model文件迁/data并保留原链接。所有文件先重验本地SHA、远端SHA及无使用者；迁移目标再次验SHA。系统盘18.83→40.96GiB，数据盘28.73→32.39GiB空闲；原始轨迹、日志、配置、评估不删，旧归档不改。受保护30SHA不变、旧38链接及新增18链接有效。对应本地完整权重保留，GitHub不含权重。严禁把按receipt移走的completed checkpoint当失败重跑。

CERT独立目录/home/vipuser/golden-cert-20261101/code实际位于/data/patr-storage/golden-cert-20261101/code。本地work/golden-cert-20261101/{TASK,PROTOCOL}.md；代码包含entropy/select/collect、G1采样及gate、G2入口及gate。train.py/patr_core.py与B2字节相同。10项CPU测试通过、配置原键与B2完全一致，G0完成28文件。GPU一致性尚未运行，不能宣称通过；相关工作表尚未作新的原始来源核验。

按用户明确授权两臂间插入，CPU原监督器146187在核验身份后退出；PaTR训练146367及start_ticks10239191不变，保留继承的supervisor.lock。新CPU等待器151967（接续记录runs/queue/handover.json）只等该训练自然结束。它随后运行PaTR supp500、请求平台实时刷新、完整CERT一致性+G1，之后再次请求平台刷新再交回原冻结B3队列运行Random smoke/400/supp500。G1失败只停止CERT后续训练，继续原Random；不能因G1失败改参数。额外seed/G2须后续时空准入，当前未启动。

平台实际刷新仍为11-02 15:09:44到期，中央截止12:09:44未改。新等待器到G1和Random长任务准入时写runs/queue/rental_check_request.json并等待waiting_platform_refresh。20分钟巡检应读取请求、刷新Galaxy UI到期时间，保存artifacts/platform_verification.json的checked_unix（晚于请求）及确实看到的expiry/cutoff；不能推定续租。GPU一致性0/1/20逐位不符即停止CERT。G1预留2h与后续Random6h等完整预算；不足则capacity_pending，不静默缩减。16GiB准入和3小时备份保留不变。

下一步：实时巡检B3和新CPU等待器，遵循PROTOCOL.md，不提前占GPU。G2及更后阶段尚未准入。
