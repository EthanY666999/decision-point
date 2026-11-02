# CERT r1执行计划

先正式两模型G1 CPU判定，完成源代码/配置审计和平台/磁盘准入；再仅参考模型新种子技术验收；通过后依次UCT400、Uniform400。保留全部旧失败证据，训练未开始前冻结清单。

## 正式G1与运行前冻结

用户新授权：仅删除经验A/B差异上限硬门槛，沿用全部固定容差；新种子仅参考sft_patr400验收；G1只用SFT/SFTGRPO400；通过后UCT→Uniform各fresh400，150只观察。旧失败不覆盖。

正式G1 gate pass_gate=true：UCT120/384，Uniform110/384，189 UID聚类bootstrap20000次seed20261101，95%CI[0.0076923076923,0.046875]；B1 K2 PaTR118组、Random112组；UCT生成37430（全probe计入），两基线35476/35788，均在115%内。参考模型不参与关卡，两个已通过技术验收的模型结果不重跑。

新目录/data/patr-storage/golden-cert-20261102-r1/code，home兼容链接。新采样种子运行前固定342153373,125988633,815617992,527987145,1729954817,620349525；不与旧6组重复。12CPU测试通过，trainer/sampler字节一致，每臂配置只steps150→400。预期训练14h，准入保留17.25h总窗及每臂8h上限，截止仍11月3日12:09:44。新CPU队列scripts/queue_400.py，先参考验收后UCT400再Uniform400，第二臂前刷新平台。无参考G1重复采样、无150 gate提前停止。

磁盘预处理：B3 Random正式10文件完整本地SHA已重新复核；拟仅移走已完成冗余checkpoint.pt（5.52GiB），仍须GitHub证据同步、远端完整SHA及无使用者复核。final_model及所有轨迹日志评估保留。

## 2026-11-02 11:52 已启动新种子参考验收

正式G1通过，运行前证据已推送2964c86。11:49实际刷新Chrome Galaxy仍到期11月3日15:09:44，checked_unix1793591367.574晚于请求1793591295.9230158；中央截止11月3日12:09:44不变。

磁盘清理完成：完整本地10文件SHA重扫、远端checkpoint SHA相符、无使用者且持有GPU锁后，仅移除已完成B3 Random正式冗余checkpoint.pt，释放5928770713字节。收据output/golden-cert-20261102-r1/disk/receipt.json；最终模型/轨迹/日志/配置/评估保留，本地完整权重优化器保留，不得误重训。数据盘26.65GiB、系统40.85GiB空闲，30保护hash不变。

CPU监督器251698运行queue_400.py；11:49:55启动新种子参考验收252128/start_ticks16439085，11:51:43完成第1组B，共6组。GPU只有此任务，18887MiB/49140MiB、36℃。旧失败result哈希未改。无新训练/验证/测试，验收通过后自动UCT400，Uniform前需新平台刷新；150只观察。

20分钟汇报已恢复ACTIVE，更新为r1队列。备用本地backup_completed_local.py --selector uct/uniform仅在complete400后做完整SHA备份，活动checkpoint不复制。本地121GiB可用。GitHub不含权重。
