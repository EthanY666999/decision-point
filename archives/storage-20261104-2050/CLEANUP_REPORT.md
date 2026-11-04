# 存储清理完成报告（2026-11-04）

已先将删除清单、评估核对和证据推送到 main（c8dae050828aa3c085e29c8d9cb36960c5356fc8），随后删除 15 个明确列入清单的大文件。
共删除 61.263 GB：5 组完成的冒烟测试权重与 checkpoint、两组 V1 已完成实验的权重与 checkpoint，以及 UCF seed42 已完成且测试核对无误的 checkpoint。

## 证据与保护

4,887 个源码与证据文件以 26 个压缩包保存；最大单包 61.71 MB。原始证据文件仍在服务器，没有删除轨迹、日志、配置、源码、评估或失败记录。2 个疑似含凭据的文档未公开，路径和哈希在 evidence_index.json 中。

原始 SFT、UCF seed42、Random K=2 seed42 的最终模型保留；正在训练的 Random K=2 seed43 checkpoint 保留。GRPO4 权重早先仅有历史 Mac 备份，本次从该已有备份传回服务器，SHA256 与历史清单一致；没有新增 Mac 备份。CERT-UCT/Uniform 额外保留。未来 seed 模型应按 STORAGE_POLICY.md 保护。

完成核验时 Random seed43 仍在训练，已至 242 步。25 个冻结源码文件哈希全部一致。

## 剩余磁盘空间

- /: 123.30 GB / 114.83 GiB 可用。
- /data: 55.15 GB / 51.36 GiB 可用。

本次为一次性安全清理。未修改、重启或暂停训练与排队调度器；未安装自动删除未来 checkpoint 的后台任务。完整策略另存于 /home/vipuser/EXPERIMENT_STORAGE_POLICY.md。

删除明细见 deletion_manifest.json；删除后逐文件磁盘状态见 cleanup_receipt.json；GRPO4 恢复证明见 grpo4_restored.json。
