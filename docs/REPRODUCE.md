# 合作者复现指南

## 1. 环境

完成实验的记录环境：Linux、Python 3.12.3、A100-SXM4 40GB、PyTorch 2.7.0、vLLM 0.9.2、XFormers 0.0.30、Transformers 4.51.3、NumPy 2.2.6、Gymnasium 1.1.1。六个核心Python依赖已固定版本；这不是含全部系统/传递依赖的容器镜像，不能承诺不同驱动/硬件逐位一致。

使用`code/requirements.txt`。训练采用XFORMERS和eager推理，配置位于`code/artifacts/runtime_override.json`；不需要安装flash-attn。服务器需要`Python.h`和C/C++编译工具；例如Ubuntu上由管理员安装`build-essential`及与Python版本匹配的`python3.12-dev`。

按原机器训练曲线，PaTR400步约7.3小时、Random约9小时，不含所有初始化/收尾时间；其他硬件不可直接套用。保守预留每方法50GB磁盘空间，空间取决于原始rollout长度和checkpoint；监控`code/runs/`。正式训练保存完整轨迹、FP32训练权重与优化器，不能只按最终模型大小估算磁盘。

本项目无需OpenAI API、外部教师模型或API密钥。安装依赖及首次下载模型需要访问对应源；也可在别处下载后复制。不要把密钥写进仓库。

## 2. 基础模型

`python scripts/prepare_model.py`下载`Qwen/Qwen2.5-0.5B-Instruct`固定revision `7ae557604adf67be50417f59c2c2f167def9a775`，随后逐文件SHA256核验。目标目录是`code/artifacts/Qwen2.5-0.5B-Instruct/`。

离线方式：将相同revision的完整模型复制到该目录，运行`python scripts/prepare_model.py --verify-only`。文件列表见`artifacts/model_manifest.json`。不要用同名模型的其他revision替代后声称完全相同实验。

## 3. 验证与训练

从`code/`执行README中的安装、CPU测试及GRPO冒烟命令。冒烟为4训练任务、1次优化更新，验证使用固定验证集前4题，收尾使用验证集第5～8题；不动正式test。`runs/smoke/`与`runs/formal/`独立。

正式训练顺序建议GRPO → 验收基线 → PaTR → Tree-Random。每种方法从base独立训练，400批、每批64题、每32题一次优化器更新，共800次更新。使用`python scripts/run.py --method METHOD`。便携入口拒绝覆盖已有目录，且同一工作区用文件锁防止并发启动。

正常完成后：

```text
code/runs/formal/patr_seed42/
  config.json                 # 实际生效配置
  source_snapshots/           # 源码快照
  train_metrics.jsonl         # 每批训练指标
  branch_events.jsonl         # 分叉/剪枝/叶子事件
  rollouts_0001.jsonl.gz ...   # 原始训练轨迹
  validation_0000.json ...    # 每40步固定验证
  final_test.json             # 400步后100题主测试
  final_test.jsonl            # 原始测试响应和动作
  checkpoint.pt              # 恢复训练需要的完整状态
  final_model/               # 最终推理权重与tokenizer
  status.json
```

不要把训练单批成功率与最终test成功率直接比较，也不要按test选最佳checkpoint。

## 4. 中断、恢复、换seed

前台`Ctrl-C`或向训练进程发送`SIGTERM`会设置停止标志；训练器在安全批次边界保存，因此可能需要等待当前采样/更新完成。不要用`kill -9`代替正常保存。远程长任务可在自己创建的tmux会话运行，电脑断网不影响服务端会话。

```bash
python scripts/run.py --method patr --resume
```

恢复要求完整`checkpoint.pt`，不能仅用`final_model/`。恢复后重新采样不保证逐位复现原进程。默认无租期截止；如需自动到期保存，可自行设置`code/artifacts/deadline.json`的Unix秒字段`unix`，它是本地忽略文件。发布包没有继承原机器的过期时间。

新seed：复制`configs/frozenlake_05b.json`为一个新配置，只修改`seed`，然后传`--config configs/your_config.json`。任务清单保持原样，输出按seed分目录。改其他训练参数属于新实验，不能称为本轮原样重跑。

## 5. 只评估已有最终模型

```bash
python scripts/evaluate.py --model /absolute/path/to/final_model --output /absolute/path/to/new_test.json
```

固定100题test、每题4条，沿用原采样参数和seed。此入口为兼容性复用原`build_models`，会加载actor/reference/vLLM副本，需要接近训练时的GPU空间；不执行梯度更新。输出路径必须不存在。重新加载/重新采样可能因数值、运行状态和调度产生与历史文件不同的轨迹，不覆盖历史结果。

新增500题结果是事后补充评估；任务清单和原始轨迹在`results/supplemental500/`。本入口仅用于原100题，不把500题替换训练中的固定test。

## 6. 只在CPU看证据

```bash
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
```

不需要下载模型。CPU检查验证代码身份、环境/解析接口与分支/权重合同，不等同于400步科学复现成功。
