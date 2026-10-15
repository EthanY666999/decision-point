# v4.2 已改好，交给 Codex 执行

目录：`/home/vipuser/patr-reproduction-v42`。它从 v4.1 实际运行的源码 `patr-reproduction-v41-candidate-b` 复制而来（sha256 已核对一致）。v4.1 的所有目录和 run 都没有改动。

## 一、已完成的修改（只有这些）

| 文件 | 改动 |
|---|---|
| `loss_v41.py` | 新增 `token_mean(values, minibatch_tokens)`：本 micro 的 token 损失之和 ÷ 整个 optimizer minibatch 的 assistant token 总数。各 micro 累加后正好等于一个 minibatch 的 token 平均（verl 默认 `token-mean`，DAPO 同款）。原 `trajectory_mean` 保留不删 |
| `update_v4.py` | PG+KL 的聚合由 `trajectory_mean(...)` 改为 `token_mean(..., minibatch_tokens)`。其余逻辑一字未动：完整题组优势、冻结 anchor/ref、clip 0.2/0.28、low_var_kl 0.001 |
| `training_contract.py` | 梯度跳过阈值由写死的 10.0 改为 `SKIP_THRESHOLD=1e5`（rLLM 默认）。非有限梯度仍然硬停；grad clip 1.0 不变 |
| `train_v4.py` | ① 配置校验改为 `loss_aggregation='token-mean'`、`grad_skip_threshold=1e5`、`training_guard_version=2`<br>② 与 R1b 配置比对时，豁免只影响训练的键（loss_aggregation、grad_skip_threshold、training_guard_version、fence_loop_alert_fraction、run_label）；采样相关的键仍必须与 R1b 完全一致<br>③ 新增每步指标 `fence_loop_fraction`、`fence_loop_alert`（>1% 时记录报警，不暂停）。现有 4 项护栏不变 |
| `config_candidate.json` | 只改 5 个键：`loss_aggregation`、`training_guard_version`、`grad_skip_threshold`、`fence_loop_alert_fraction`、`run_label` |
| `tests/test_v42.py` | 新增 7 项测试：token 等权、micro 累加等价、万字循环不被降权、总梯度有界、阈值 1e5 不跳过、非有限硬停、循环检测 |
| `tests/test_optimizer_skip_v41.py` | 显式传入 `skip_threshold=10.` 来测旧规则 |

## 二、已验证
- `/home/vipuser/patr-reproduction/.venv/bin/python -m unittest`：test_v42、test_optimizer_skip_v41、test_v41_contract、test_driver_monitor_integration、test_guard_checkpoint，共 16 项全部通过（CPU，`CUDA_VISIBLE_DEVICES=` 为空）。
- 用循环检测器回放 v4.1 run b 的真实 rollout，循环回合比例如下：

  | 批次 | 10 | 30 | 50 | 54 | 55 | 56 | 57 |
  |---|---|---|---|---|---|---|---|
  | 比例 | 0.04% | 0.48% | 0.11% | 0.12% | 0.91% | 0.98% | 2.53% |

  第 57 批会触发报警，比 5% 截断护栏早一批。
- **未做**（需要 Codex 按自己的验收流程完成）：GPU 上的 Qwen 更新验收、`frozen_identity.json`、`reports/r0_gate.json`。v42 目录里目前没有这两个文件，`train_v4.py` 会拒绝启动，这是有意的。

## 三、Codex 需要做的步骤

1. 在 v42 目录按原流程重新生成 GPU 验收：`qwen_update_acceptance.py` 和 `update_integration_acceptance.py`，覆盖新的 token_mean 路径和恢复路径。通过后生成 `frozen_identity.json` 和 `reports/r0_gate.json`。不得复用 v4.1 的 gate。
2. R1b 打分选择沿用 `/home/vipuser/patr-reproduction-v41-r1b/runs/r1b-a/summary.json`（map_max）。采样相关源码与配置没有改变，所以不需要重跑 R1b。
3. 从 base 模型新开 PaTR，先跑 80 步：
   ```
   cd /home/vipuser/patr-reproduction-v42
   /home/vipuser/patr-reproduction/.venv/bin/python train_v4.py --phase patr \
     --out runs/patr-400-v42-a \
     --r1-summary /home/vipuser/patr-reproduction-v41-r1b/runs/r1b-a/summary.json \
     --stop-at 80
   ```
4. 80 步内护栏没有触发，就用同一指纹 `--resume --stop-at 400` 继续，最后做 test。
5. PaTR 完成后，剩余时间不少于 14 小时才跑 GRPO（同配方）。截止仍为 10 月 17 日 05:00。

## 四、遇到这些情况，停下来报告用户，不要自行处理
- 任一护栏暂停（token_limit 超过 5%、长度超过 3 倍、梯度超过 20 倍、耗时超过 4 倍）。
- `fence_loop_fraction` 连续 3 批超过 1%（只报告，训练不停）。
- 兜底方案「单回合上限 1024 token」需要用户同意后才能启用。
- 不从 v4.1 的任何 checkpoint 或 history 权重续跑或热启动；不看 validation 或 test 来改配置。

## 五、在报告中标注的来源
- token-mean：偏离论文 Eq.2。依据是 v4.1 第 55–58 批的围栏循环证据：Eq.2 下一条万字循环轨迹的每个 token 只分到约 1/2500 的权重。v4 token-sum 失败的证据也一并引用。
- 梯度跳过阈值 1e5：rLLM 配置默认值。阈值 10 只在 token-sum 尺度下才有意义；v4.1 在 Eq.2 尺度下因此误跳过了第 17、26、31、32、44 步的更新。
