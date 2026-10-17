# 发布验收：2026-10-17

## 已完成

- 全新本地Python 3.12虚拟环境安装`requirements-cpu.txt`成功，4项CPU检查通过；A100服务器上同样4项通过。
- 核对远端当前正式实验的`train.py`与`patr_core.py` SHA256，与发布代码及冻结归档逐字一致。数据、配置、runtime override和vendored Python文件通过`SOURCE_MANIFEST.json`校验。
- 固定Qwen revision的9个模型文件在A100端逐文件SHA验证通过。
- **使用新`scripts/run.py`入口在A100单独临时目录完成GRPO冒烟**：初始化验证、4题×8轨迹采样、1次真实优化更新、权重同步、checkpoint保存、收尾验证、最终模型导出，进程退出码0，状态`smoke_complete`。参数最大变化约1.0133e-6，checkpoint约5.93GB，FP32最终模型约1.98GB。没有启动新400步实验。
- 冒烟只使用验证题，不触碰正式测试题；短测试采用1024响应token上限，因此存在截断和低成功率，它验证运行链路，不证明模型效果。日志末尾有未显式销毁NCCL进程组的退出警告，但退出码0、导出完整、进程已结束。
- 冒烟指标和日志在`results/release-validation-20261017/`。确认完成后只清理本次临时smoke权重/checkpoint，原正式运行未修改。
- `scripts/summarize_results.py`重算100题与500题的三方法均值、配对bootstrap区间，与已存报告一致。
- 所有发布Python文件语法解析成功；新入口帮助可运行；未发现常见API token或私钥模式；不提交模型权重、虚拟环境或本机截止时间。

## 验证边界

A100验证复用了原已安装的GPU Python环境，未在全新Linux机器重新安装所有GPU依赖。核心6个依赖版本与原环境一致，传递依赖及系统驱动未全部锁定，安装说明要求`pip check`。未在此轮打包重新执行400步训练，也不声称换机可逐位重现历史曲线。

此次整理不修复算法或数据问题；首回合无效动作、单seed、地图几何重叠和论文实现细节不确定性仍需如实报告。

当前发布文件完整性可在任意安装Python的机器运行：

```bash
python code/scripts/verify_release.py
```

`RELEASE_MANIFEST.json`记录除自身外的发布文件哈希和大小；代码冻结身份另由`code/SOURCE_MANIFEST.json`约束。
