# 2026-11-03 合作者 main 发布说明

## 发布范围

将研究分支 `a53d840b9fa10b19bf25849d431d115d36678371` 合入 main 原版本 `cf6391404c373f92ccfafd13fdde6af2caa5acc7`，保留历史结果与来源。朋友的唯一任务入口为[精简任务书](PATR_3B_4B_REPRODUCTION_TASK_2026-11-01.md)。旧分支不删除、不改写历史，但不再要求合作者从它获取代码。

- 主线：两个模型各三种无SFT RL、一次SFT、三种SFT后RL，共12次RL和2次SFT；已完成跳过。
- 本轮不新增SWE、无SFT GRPO4、K扫描/map8/CERT/DAPO/ARPO。SFT GRPO8和额外seed暂不排队。
- 当前正常运行先完成，失败尝试保留；未因这次整理操作任何远程训练。
- [内网更新说明](COLLABORATOR_UPDATE.md)：只拉main，不要求上传或push；有本地修改和运行中的工作区先保留。

## 工程变更与科学口径

1. main现有SFT、补500评估及原实验分支的全部已发布文件，不需跨分支拼装。
2. train.py新增配置项optimizer=adamw8bit，默认仍是原AdamW及原参数。8-bit是可选数值实现差异，不伪称与原优化器完全等价；已完成历史训练不受本次改动影响。SOURCE_MANIFEST.json同时保留historical_files和当前files哈希。
3. sft.py提供--optimizer；audit-only仅依赖tokenizer和CPU环境，可在无torch的审核环境完成。正式SFT共享工作区锁，仍不提供断点恢复。
4. 两种评估入口支持safetensors索引及分片完整性检查。--runtime-config只传工程字段，保留原评估题集、解码与种子；不能拿config.json存在代替权重完整性。
5. prepare_collaborator.py生成不覆盖既有不同内容的本地配置；signal_stats.py解析完整JSON而非正则首个reward，缺少201–400批时明确标记。
6. Golden Trace原始数据、环境源码、patr_core.py、冻结0.5B配置和已归档结果保持原值。历史EXPERIMENT说明顶部加入入口提示，不重写原执行日志。数据README更新为当前main入口，数据文件哈希未变。
7. 新增配置/模型/日志的gitignore条目，避免把内部运行材料误加入代码仓库。文件不会因此被删除，也不要求同步到GitHub。

## 本地验证

环境：macOS arm64、Python3.12；新建隔离CPU环境，numpy2.2.6、gymnasium1.1.1、transformers4.51.3、jinja2 3.1.6。首次模板检查发现独立CPU环境缺jinja2，补齐后重新执行通过。

- 通用CPU测试：8项通过；无SFT_TEST_MODEL时2项模板测试明确跳过，随后对两个指定模型分别执行2项模板测试并通过。
- 两个真实tokenizer分别审核完整1,000条SFT消息：均19,676监督token，最长2,376 token；精确前缀、目标动作及结束token断言通过。
- tokenizer revision：Qwen2.5-3B-Instruct `aa8e72537993ba99e69dfaafa59ed015b17504d1`；Qwen3-4B-Instruct-2507 `cdbee75f17c01a7cc42f958dc650907174af0554`。这里只下载tokenizer/配置，未下载完整模型权重，不能将其称为GPU模型验收。
- Golden Trace：1,000条逐条环境回放、地图唯一性、完整/SFT消息对应关系通过。
- 补充500题：任务文件哈希和5个固定批次分区校验通过。
- signal_stats 对现有 B3 PaTR 的400个真实rollout批次进行只读重算，与每批日志的mixed_group_fraction逐项完全一致。
- 新增检查覆盖：缺分片及不安全索引拒绝、评估工程参数不覆盖解码/seed/采样次数、配置覆写拒绝、嵌套过程reward不污染最终奖励统计。
- 发布前校验当前文件清单、Python语法、当前入口文档链接与代码围栏；不把旧历史路径当作当前运行入口。

**未验证：本次没有运行3B/4B GPU训练或实际8-bit GPU优化器步骤，没有证明显存档位一定适合朋友的A100。**首个GPU运行仍需在目标机器做前10批资源与参数更新验收；不能据CPU检查宣称完整复现成功。

## 版本校验

RELEASE_MANIFEST.json覆盖当前发布文件（自身除外）；历史快照中的服务器符号链接只校验链接文本、不解引用，也不是可移植运行依赖。GOLDEN_SYNC_MANIFEST.json保留原同步范围并更新当前文件哈希、标记此次main发布。旧版本清单仍可通过Git历史读取。历史目录中嵌入的源码清单与执行身份保持原样。

```bash
python code/scripts/verify_release.py
python golden-trace/verify.py
```
