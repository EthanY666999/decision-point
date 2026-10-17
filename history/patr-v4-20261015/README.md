# PaTR v4执行交接（2026-10-15）

用户授权：先迁移旧大文件至数据盘，轻量备份后按R0→R1→烟测→PaTR400→GRPO400执行。

当前训练停在R0来源/等价性门槛，尚未启动正式训练。详见[R0说明](R0来源验收与暂停说明.txt)和[数值证据](r0_equivalence_report.json)。

- [最终方案](PaTR_v4最终实验方案.txt)
- [执行授权](转发给复现实验纠错的执行命令.txt)
- [来源与阶段状态](source_and_stage_status.json)
- [历史真实PaTR叶子数量](real_patr_group_sizes.json)
- handoff/v2、handoff/v3：既有源代码、配置、逐步指标和诊断，属于历史交接，不是v4正式运行源码。
- upstream_*：按指定提交提取的来源文件，用于核对语义；上游仓库许可证适用。
- 大权重、optimizer、完整token轨迹和归档位于本云实例/data，未进入普通Git历史。
- 同机数据盘不属于独立异地备份，租期结束后的保留规则尚未核实。

迁移完成：约60.6 GiB已按SHA256核验后释放系统盘源副本。原路径保留链接，清单见[migration manifest](migration_v4_manifest.json)。系统盘约71 GiB、数据盘约38 GiB可用。
