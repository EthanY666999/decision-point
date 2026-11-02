# 2026-11-03 深度清理与历史恢复

本轮依据用户再次腾出双盘空间、保留完整溯源能力的要求，移走服务器上已验证存在完整副本的权重。原始 SFT、运行环境、新项目、完整轨迹、代码、配置、日志、评估和失败证据保留。

- `pre_cleanup.json` 给出每个源路径、实际路径、大小、SHA256、备份位置和删除前状态；`receipt.json` 是最终执行收据。
- `local_candidates.json` 包含 19 组完整本地包的逐文件 SHA256；所有文件在本次重新扫描过。`local_backup` 为 Mac 上现存的确切文件路径。权重不进入普通 Git 仓库。
- `archive_members.json` 为完整扫描的旧归档成员索引（60942 个成员，大文件逐个 SHA256）；原归档保留在 `/data/patr-storage/patr-archives/patr-user-stop-20261014.tar.gz`。归档整包 SHA256 在删除前重新验证。
- `light_runs/` 为清理前保存的非权重运行记录；Git 中 branch_events.jsonl 以无损 gzip 保存，原文与校验收据仍在本地 output。
- 每个被移走的权重旁新增 `文件名.ARCHIVED.json`，说明该文件的确切恢复来源。该索引不是可加载的模型；使用前需要恢复权重。

## 恢复

对于 `backup_kind=local_complete_package`，从 `local_backup` 指定的 Mac 文件通过 rsync 复制到服务器原路径。其他 tokenizer/config 文件仍在原目录；完整包也保存在 `packages[].local_root`。恢复后用 sha256sum 核对 manifest 中的 sha256，确认成功再加载。

对于 `backup_kind=retained_remote_archive`，用 Python tarfile 打开 `archive`，精确获取 `archive_member`，将 extractfile 返回的字节写入原 `source` 路径；只恢复指定成员，不要将整个归档覆盖解压到运行目录。恢复后核对大小和 sha256。

旧归档不是完整本地备份，因此继续保留远端原件。未验证的旧实验独有权重、本机环境和新工作目录都不在清理范围。删除清单代表有备份的文件归档，不代表训练失败，不应因此重训。
