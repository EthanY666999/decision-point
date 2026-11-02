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

## 2026-11-02 12:19 验收通过，修复日志字段冲突后恢复UCT

新6组参考验收passed=true，全部固定CI容差与1922节点校准通过，原失败证据不改；不重跑验收或G1。验收累计非训练成本：输入16208557、生成82432、环境动作19644（无效731），总wall627.08s；其中probe输入3100396、生成3844、0环境动作、4.351s，已计入总成本。

首次UCT于12:03:43在第1次更新后写record时TypeError退出：collector和optimizer均提供sampled_surprisal。无train_metrics、无checkpoint，不能宣称完成1步。完整日志/状态/初始验证及源manifest已保存在本地recovery_20261102_1216和远端runs/recovery_20261102_1216，失败UCT目录已归档。只改run_cert.py记录适配，将collector旧概率诊断重命名rollout_sampled_surprisal，保留optimizer的sampled_surprisal；train/patr_core/cert_collect及配置、采样、奖励、优化器、验收阈值均不改。原12CPU测试和新增字段冲突回归共13项通过，manifest更新保留旧版。

确认GPU空、原监督器退出并取得两把锁后恢复；因首次尚无checkpoint，从同一SFT fresh恢复，非伪造--resume。新监督器258947；12:19:13启动UCT PID259328/start_ticks16614871。新参考验收复用不重跑。12:18实际刷新Chrome平台仍11月3日15:09:44，checked_unix1793593117.707晚于新请求1793593093.7909138，中央截止不变。Uniform前仍须另一次刷新。SSH偶发认证前Connection closed，重试可恢复；不能因此重启训练。最新已确认完成训练步基线仍0，等待首步持久记录。
