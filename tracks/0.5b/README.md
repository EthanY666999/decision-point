# 0.5B 历史实验与归档

模型为 Qwen2.5-0.5B-Instruct。本入口不启动训练，不将缺失对照标为已完成。
用户已决定不再续跑0.5B；旧监控已由原聊天停用。未完成项仍按未完成留档，不自动重启。

## 固定快照与活动证据

- [整理前 main 快照](https://github.com/EthanY666999/decision-point/tree/archive/0.5b-20261108)：指向 `fd573a6c08941b0260a7ffc260a6903257eb7cb1`，保留当时所有文件；是快照，不意味着服务器全部文件已归档。
- [历史 Golden Trace 分支](https://github.com/EthanY666999/decision-point/tree/codex/golden-sft-patr-20261031)：保留旧链接，不作为当前 2B 入口。
- main 的 `archives/` 仍可接收经过核验的收尾证据；不覆盖固定快照。

## 按实验阶段查找

| 阶段 | 路径 |
| --- | --- |
| 原 PaTR / GRPO / Random 复现 | [主100](../../results/main100/)、[补500](../../results/supplemental500/)、[原指南](../../docs/REPRODUCE.md) |
| Golden Trace SFT 与控制组 | [SFT结果](../../results/golden-sft-20261031/)、[实验协议](../../experiments/) |
| SFT-UCF、Random、消融及源码证据 | [11月4日证据包](../../archives/storage-20261104-2050/)（先读 evidence_index.json） |
| 无 SFT UCF | [无SFT证据包](../../archives/ucf-nosft-20261106/) |
| UCF seed44、无SFT UCF/Random累计结果 | [扩展归档](../../archives/ucf-extended-20261106/) |
| V1–V4及纠错历史 | [历史索引](../../history/README.md) |

`main/code/` 是便携基础训练器，不代表每一轮 UCF 的完整源码；复核时使用对应归档的源码快照、配置和哈希。
轻量报告与原始逐题结果分开；主100、补500、filtered409、map8 不混用。缺失 seed/未完成实验按实情标注。
模型权重按原存储策略留服务器，本入口不复制权重到本地。

## 已核实的归档缺口（2026-11-08）

- Random K2 seed44最后确认62/400步，无最终测试及完成归档；部分记录可能仍在到期服务器。
- noIS43、uniform43和StarPO-S/GiGPO四项外部基线未运行；tau42/43已取消。
- UCF43、Random43及部分消融已有汇总数字，但完整原始记录的归档覆盖尚未逐项确认，不写成“全部备份齐全”。
- 外部基线准备源码/CPU验收未在已发布包内确认。恢复服务器访问后可以只补证据，不因此重跑。
- 无SFT UCF的原归档与扩展补充归档是同一次实验的两部分，均保留。
- archives内的extensions_v2/v3是调度层版本，不等同于2B科学算法UCF-v2。
- 控制器振荡、seed43主100下降、无SFT UCF落后Random等负面结果保留。

累计结果和seed44源码入口：[完成时累计报告](../../archives/ucf-extended-20261106/ucf_seed44/report_at_completion.json)。
