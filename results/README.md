# 结果与日志

- `main100/runs/patr_seed42/`、`main100/runs/tree_random_seed42/`：当前400步训练指标、验证和主测试原始轨迹。
- `main100/v1_baselines/`：既有GRPO与旧树模型结果，仅供对比，不是当前源码。
- `main100/analysis/REPORT.txt`：主结果、置信区间、格式和树策略审计、map8附加口径。
- `main100/reports/`：本轮执行日志、完成记录、诊断。`stage1`关卡失败报告是更早阶段的观察记录，之后用户明确授权继续正式训练；完成状态以`sequential_status.json`与两份completion为准。
- `supplemental500/REPORT.txt`、`results.csv`：500题补充结果和解释。
- `supplemental500/{grpo,patr,tree_random}/`：分批评估证据；`tasks.json`/`maps.json`保存固定题集。

此处是只读证据，含历史服务器路径/时间，不应当成可执行脚本。`MANIFEST.json`、`REMOTE_ARCHIVE_RECEIPT.json`等保留的是整理前原始归档凭据，可能列出此次移除/迁移的源码，不作为当前目录完整性清单；当前发布文件校验用根目录`RELEASE_MANIFEST.json`。

完整800个训练rollout和大模型权重没有在Git中。`raw_rollout_manifest.json`只证明当时记录的文件身份，并不表示那些文件可从本仓库下载。权重备份收据只说明当时备份完成，不提供公开下载地址。
