# 3B / 4B 运行代码

从仓库根目录的 README 开始，再读 ../docs/PATR_3B_4B_REPRODUCTION_TASK_2026-11-01.md。
使用 scripts/prepare_collaborator.py 生成各模型的独立本地配置，不修改跟踪的模板。
train.py、patr_core.py、sft.py、环境和数据沿用发布前原值；SOURCE_MANIFEST.json记录身份。
本分支不包含完整历史结果，summarize_results、golden_compare等历史统计脚本并非本次交付入口。
运行所需的补500协议及批次校验文件仍保留在 results/supplemental500/。
