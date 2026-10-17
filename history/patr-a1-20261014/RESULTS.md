# 本轮实测结果

更新时间（UTC）：2026-10-14T10:08:27.948836+00:00

复现类型：paper-based reimplementation；完整假设与差异见 [REPRODUCTION_AUDIT.md](REPRODUCTION_AUDIT.md)。

## 正式训练

| Method | 状态 | 步数 / 400 | Success % | Pass@4 % | RLen | Turns | 已记训练小时 | 最近每步秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| grpo | complete | 400 | 61.00 | 62.00 | 17.67 | 4.42 | 11.54 | 90.32 |
| dapo | stopped_by_user | 5 | — | — | — | — | 1.65 | 1233.11 |
| arpo | pending | 0 | — | — | — | — | 0 | — |
| tree_random | complete | 400 | 43.00 | 45.00 | 18.89 | 4.72 | 14.15 | 168.53 |
| patr | complete | 400 | 60.00 | 61.00 | 18.82 | 4.71 | 8.17 | 62.44 |

未完成 400 步时不填最终测试指标。表内训练小时不含首次模型加载、首次验证或回滚后废弃的计算；总墙钟成本另见队列日志。

grpo 另有 0.227 小时、76958 个生成 token 的已完成迭代因检查点回滚而废弃。该开销必须计入复现成本，且尚不含触发故障的未完成迭代。证据保留在 `recovery_history/`。
tree_random 另有 0.091 小时、200916 个生成 token 的已完成迭代因检查点回滚而废弃。该开销必须计入复现成本，且尚不含触发故障的未完成迭代。证据保留在 `recovery_history/`。

## 已完成方法的最终动作质量

| Method | 无效动作 % | 成功轨迹 / 400 | 失败或截断分类 |
|---|---:|---:|---|
| grpo | 22.64 | 244 | hole: 130, turn_limit: 26 |
| tree_random | 21.18 | 172 | hole: 204, turn_limit: 24 |
| patr | 42.56 | 240 | hole: 160 |

无效动作比例以实际执行动作总数为分母；一条最终成功的轨迹也可能包含无效动作。GRPO 后期出现 Goal 等非法指令，PaTR 后期出现代码块结束符不完整；原始解析器与奖励保持不变。训练行为及格式审计见 [REPRODUCTION_AUDIT.md](REPRODUCTION_AUDIT.md) 和 `reports/*action_format_audit.json`。验证集成绩不替代本表的最终测试。

## 链路烟测

| Method | 状态 |
|---|---|
| grpo | smoke_complete |
| dapo | smoke_complete |
| arpo | smoke_complete |
| tree_random | smoke_complete |
| patr | smoke_complete |

烟测为 1 步小批次，仅检查真实训练链路；不可用于方法效果排序。首次 GRPO 烟测曾评估 4 个 test 任务，后续烟测改用 validation 任务；未以该结果选择算法超参。

## 可审计文件

- `reports/summary.csv`：可直接用于表格。
- `runs/formal/<method>_seed42/train_metrics.jsonl`：逐步真实采样、训练、成本。
- `runs/formal/<method>_seed42/validation_*.json`：独立验证曲线。
- `runs/formal/<method>_seed42/final_test.json`：最终 100 × 4 测试。
- `runs/formal/<method>_seed42/rollouts_*.jsonl.gz`：原始轨迹、mask、采样 logprob 与动作。
- `artifacts/map_audit.json`：地图重复与最短路审计。
- `logs/queue.log`：队列执行和错误。

已有风险：上游生成器允许最短路为 11 的少数地图，但交互上限为 10；小地图存在跨 split 的重复几何结构。这两点保留并报告，未根据测试表现修改数据。
