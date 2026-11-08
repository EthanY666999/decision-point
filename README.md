# FLH Golden Trace — GPT-6.1 SOL，2026-11-08

这是为**后续难版 FrozenLake（FLH）SFT 实验**独立制作的数据。目标为 1,000 条成功教学轨迹；实际交付数量、复验结果和 SHA256 以 `summary.json` 及 `independent_validation.json` 为准。

**本数据没有替换正在进行的 2B 实验数据，也没有改变其代码、参数、评估集或训练队列。** 它与旧版 FrozenLake Golden Trace、SWE Golden Trace 和 `collab/3b-4b` 协作实验分开存放。生成数据本身不代表完成 SFT，更不代表新方法指标提升。

## 环境与选题

- 来源：2B 难环境工作线冻结的 FLH 训练地图，源码快照来自 `e32ac58b8a857eb98023ec1f79bbe0cf855c95ac` 对应工作目录；环境源文件 SHA256 逐个记录于 `manifest.json`。
- 地图边长 9–12；随机起点与终点；精确最短路 7–16 步；回合上限 20；`is_slippery=False`。
- 直接把冻结地图作为 `desc` 传入环境，不调用旧随机地图生成器；源文件中历史遗留的 `is_valid` / `MAX_STEPS=5` 不参与本次选图或回放。
- 从 10,000 张训练地图按 seed `20261108` 打乱后依次尝试，保留成功轨迹，直到 1,000 条。训练地图身份与哈希见 `train_map_hashes.json`。
- 排除 validation 100、test 100、supplemental 500，共 700 张地图。排除表为 `heldout_map_hashes.json`，不把评估地图交给教师。
- 成功筛选会产生选择偏差，难题失败不能隐去；所有失败尝试保存在 `failed_episodes.jsonl.gz`，成功样本的尺寸、路径长度分布见汇总。

## 教师与生成协议

教师通过 Codex CLI 显式请求 `--model gpt-6.1-sol`，推理档位 `medium`，禁用 shell、工具、搜索及子代理，无模型回退。实际最终回答、请求地图、token usage 和远端原始调用哈希存于 `teacher_calls.jsonl.gz`。

CLI 的模型参数和成功调用可以核验；CLI 事件未必另外给出服务端模型身份，因此这里记录的是**显式请求的模型**，不虚构额外的服务端身份认证。

**这次采用“完整动作计划 → 真实环境逐步回放”协议**：教师一次看到每张完整静态地图并输出一串 Up/Down/Left/Right；随后在冻结环境中逐步执行。与旧版每回合重新请求教师的协议不同，不应在论文里写成相同的数据生成过程。环境是完全可观测、确定性的，因此可以离线执行完整计划并记录各步反馈。

- 没有向教师提供 BFS 答案或最短路线；BFS 仅用于复核已有地图的距离约束。
- 未修正或补写教师路线；掉洞、未到终点、终止后仍有动作或非法计划均不计入合格数据。
- 每条成功路线至少在实际环境中执行两次，最终整批再回放；另附独立逐格检查器，与实际环境结果对照。
- 导出是 **action-only 教学数据**。assistant 每回合仅输出一个 fenced action，例如 ` ```Down``` `；没有伪造教师思维链或步骤解释。
- student 对话沿用原版 `FrozenLakeAgent` 提示词。该历史提示词含 slippery 的通用文字，但实际运行环境为确定性。此处为保持原实验输入格式而保留，不能据此声称本数据包含随机滑动。

## 文件

| 文件 | 内容 |
| --- | --- |
| `golden_traces.jsonl.gz` | 合格完整轨迹：原地图、任务身份、教师动作、逐步状态/奖励/观察、对话 |
| `sft_messages.jsonl.gz` | 每行一条完整 episode 的 `messages`，外加 `task_uid`、`map_hash` |
| `failed_episodes.jsonl.gz` | 全部失败尝试及实际执行结果 |
| `teacher_calls.jsonl.gz` | 教师请求、实际最终输出、usage、调用哈希；包含失败路线的回答 |
| `manifest.json` / `pilot.json` | 生成协议、源文件哈希、首批 32 张试验结果 |
| `summary.json` | 合格数、尝试数、分布、调用统计和发布文件 SHA256 |
| `independent_validation.json` | 独立几何检查与冻结环境回放的最终结果 |
| `token_lengths_qwen35_2b.json` | Qwen3.5-2B 原生非思考模板下的完整 episode 长度分布 |
| `source/rllm_source/` | 冻结环境与 agent 源码，不含模型权重 |
| `verify_release.py` | 不调用教师、不使用 GPU 的独立复验入口 |
| `remote_evidence_inventory.json` | 留在远端的原始 CLI 证据路径、大小、SHA256 |
| `remote_dataset.py` / `teacher_bridge.py` / `response_schema.json` / `package_release.py` | 本次生成器与发布器快照；内含当时机器路径，重新生成需先适配路径 |

## 下载与复验

此数据使用独立分支，建议在服务器新目录下载：

```bash
git clone --depth 1 --single-branch --branch data/flh-golden-trace-20261108 \
  https://github.com/EthanY666999/decision-point.git flh-golden-trace
cd flh-golden-trace
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python verify_release.py .
```

成功输出应包含 `verified: 1000`、`heldout_overlap: 0`、`geometry_and_environment_agree: true`。复验会重新检查全部压缩文件哈希、地图身份、唯一性、最短路范围、逐步动作和导出对话。

## 后续 SFT 使用

每行是一个完整多回合 episode，实际动作数不固定。读取 gzip JSONL 的 `messages` 字段；loss 只覆盖 assistant 回复，system/user 不参与监督。必须使用目标模型的原生 chat template，不要直接把所有 role 拼接成普通文本。按完整 episode 划分及统计，避免同一地图拆成多条后泄漏到验证集。

这些对话包含逐回合累积历史，**部分会超过 4,096 tokens**。完整 1,000 条的实测长度见 `token_lengths_qwen35_2b.json`；采用其他 tokenizer 或模板时需要重新测量。训练前应检查目标 tokenizer 的实际长度并制定独立的数据截断协议，不能无记录地裁掉后半段动作。本文不预设 SFT epoch、学习率、batch size、训练步数或新的主实验结果。

若要比较 SFT→GRPO、SFT→PaTR、SFT→UCF，各臂应从**同一个本数据训练出的 SFT checkpoint**独立开始，另保留无 SFT 对照；不要把已有旧 SFT 结果与本数据结果当作仅改变采样器的对照。

## 存储

数据、完整轨迹与原始教师调用留在服务器 `/home/vipuser/golden-flh-trace-20261108`。Mac 只保留小脚本、schema 与状态；教师最终回答经内存转发到服务器，没有在 Mac 备份数据集或权重。生成期间使用 Mac 的已登录 Codex 做教师调用转接，因此生成并非完全脱离 Mac。数据回放只使用低优先级 CPU。

源环境保留原文件的 Gymnasium / RAGEN 来源说明。共享数据不代表变更上游代码的许可；使用或再分发上游实现时应遵守其原许可。
