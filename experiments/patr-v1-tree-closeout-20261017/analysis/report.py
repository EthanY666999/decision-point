import json,csv,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/home/vipuser/patr-v1-tree-20261016');out=root/'reports/closeout-20261017'
load=lambda p:json.loads(p.read_text())
a=load(out/'statistical_analysis.json');tr=load(out/'training_analysis.json');tree=load(out/'tree_training_audit.json');maps=load(out/'map_comparison.json');integ=load(out/'integrity.json')
assert load(out/'evaluation_status.json')['phase']=='complete'
changed={m['uid'] for m in maps['maps'] if m['changed']}
matching={}
for m in ['grpo','patr','tree_random']:
 p=(Path('/home/vipuser/patr-reproduction') if m=='grpo' else root)/'runs/formal'/f'{m}_seed42/final_test.jsonl'
 old={(r['task']['uid'],r['lineage']):r for r in map(json.loads,p.read_text().splitlines())}
 new={(r['task']['uid'],r['lineage']):r for r in map(json.loads,(out/f'{m}_map8.jsonl').read_text().splitlines())}
 mismatch=[]
 for k in old:
  if k[0] in changed:continue
  if old[k]['trace']!=new[k]['trace'] or old[k]['response_tokens']!=new[k]['response_tokens'] or old[k]['reward']!=new[k]['reward']:mismatch.append(k)
 same_keys=[k for k in old if k[0] not in changed]
 reward_diff=sum(int(new[k]['reward']>0)-int(old[k]['reward']>0) for k in same_keys)
 matching[m]=dict(unchanged_map_trajectories=368,different_trace_count=len(mismatch),success_count_difference_on_unchanged_maps=reward_diff,different_trace_keys=mismatch)
 if m!='grpo':assert load(out/f'{m}_map8.json')['model_sha256']==integ['models'][m]['final_model_sha256']
(out/'supplement_consistency.json').write_text(json.dumps(matching,indent=2)+'\n')
lines=['V1 新树 PaTR / Tree-Random seed42 本轮收尾报告','日期：2026-10-17（北京时间）','',
'结论：两种新树方法均完成400个训练批次、800次AdamW更新及最终test。沿用V1 GRPO基线，无额外训练。主测试中PaTR与Random成功率接近，当前证据不足以认定PaTR优于Random，也不足以认定二者等效。PaTR的Pass@4点估计较高、无效动作较少。','',
'一、主测试（V1原测试地图；100题，每题4次；最终400步模型，不挑最佳checkpoint）',
'方法 | Success% | Pass@4% | 无效动作% | 平均回合 | sampled tokens | canonical text tokens']
for label,k in [('V1 GRPO','grpo'),('新PaTR','patr'),('新Random','tree_random'),('旧V1 PaTR','old_patr'),('旧V1 Random','old_tree_random')]:
 r=a['results'][k];lines.append(f"{label} | {r['success_pct']:.2f} | {r['pass4_pct']:.2f} | {100*r['invalid_action_fraction']:.2f} | {r['turns_mean']:.4f} | {r['sampled_assistant_tokens_mean']:.4f} | {r['canonical_decoded_text_tokens_mean']:.4f}")
lines+=['','每个task UID作为配对重采样单位，保留该题全部4条轨迹；50000次bootstrap，分析随机种子42，95%百分位区间。区间仅反映测试题抽样不确定性，不包含训练seed差异，未作多重比较校正；不同UID可能有相同地图几何，独立题假设仍是限制。']
for pair in ['patr minus tree_random','patr minus grpo','tree_random minus grpo','tree_random minus old_tree_random']:
 d=a['pairs'][pair];lines.append(f"{pair}: Success差 {d['success_pp']['difference']:+.2f}pp, 95%CI {d['success_pp']['ci95']}; Pass@4差 {d['pass4_pp']['difference']:+.2f}pp, 95%CI {d['pass4_pp']['ci95']}。")
lines+=['','二、附加地图口径（不训练，不据此改变配置）',f"对同100个UID把地图生成max_steps从10改为8，确实有{maps['changed']}张地图改变；交互上限保持10轮，提示词、采样温度0.7、top_p0.8、top_k20、每题4样本不变。模型均为既有最终权重，原测试不覆盖。",'上游来源：agentica-project/rllm 固定提交7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4 的 examples/frozenlake/train_frozenlake_agent.sh，env.env_args.max_steps=8，agent.max_steps=10。已读取上游原文核实。','https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/examples/frozenlake/train_frozenlake_agent.sh','这只对齐该地图参数，不能称为论文官方完整复现；作者实际数据和完整训练实现未核实。','方法 | Success% | Pass@4% | 无效动作%']
for label,k in [('V1 GRPO','grpo_map8'),('新PaTR','patr_map8'),('新Random','tree_random_map8')]:
 r=a['results'][k];lines.append(f"{label} | {r['success_pct']:.2f} | {r['pass4_pct']:.2f} | {100*r['invalid_action_fraction']:.2f}")
d=a['pairs']['patr_map8 minus tree_random_map8'];lines += [f"附加口径PaTR−Random：Success {d['success_pp']['difference']:+.2f}pp，95%CI {d['success_pp']['ci95']}；Pass@4 {d['pass4_pp']['difference']:+.2f}pp，95%CI {d['pass4_pp']['ci95']}。",'92张未换地图共368条轨迹的复核：'+json.dumps({k:{x:v[x] for x in ['different_trace_count','success_count_difference_on_unchanged_maps']} for k,v in matching.items()},ensure_ascii=False),'虽然使用相同逐轨迹采样种子，重新加载模型后重新采样仍与训练进程内的原始测试存在轨迹差异。执行顺序、数值计算或推理随机状态等具体来源尚未逐项隔离。因此这是一套独立的附加评估，不能把主测试到附加测试的全部增幅归因于8张地图变化，也不将附加结果替换主结果。','',
'三、长度、输出有效性与多样性',
'sampled：实际采样assistant token数（包括采样终止token），不含prompt和环境观测。canonical text：逐回合将解码后的assistant文本重新分词，不加特殊token；这是文本重分词统计，不是改变训练token路径。两者不能未经核实就视为论文RLen。',
'本轮主测试每条轨迹的sampled与canonical差值约等于回合数，符合每回合一个终止token的差异。最终输出主要是极短动作，短长度不只是统计口径造成的。',
'无效动作按原始严格三反引号解析器计算，不修复格式、不重新给分。环境将无效动作作为原地停留，消耗一轮，之后仍可成功；因此Success不是输出格式正确率。全部主测试和附加测试轨迹均以原环境重放，动作解析、逐步状态、奖励和总奖励一致。']
for label,k in [('V1 GRPO','grpo'),('新PaTR','patr'),('新Random','tree_random')]:
 r=a['results'][k];lines.append(f"{label}: 4条动作序列完全相同的题占{r['all_four_same_action_sequence_pct']:.0f}%，平均不同动作序列{r['mean_distinct_action_sequences']:.2f}；无效输出分类{r['invalid_categories']}。")
lines+=['输出格式补充审计：V1 GRPO、新PaTR、新Random的主测试首回合均400/400为无效动作。GRPO常输出非动作符号（如三反引号内的G/P），PaTR首回合缺少闭合围栏的一个反引号，Random常输出ACTION文本或非动作符号。PaTR与GRPO的400次无效均在首回合；Random还在第2/3/5回合出现74/52/2次无效。原环境反馈后仍可继续成功，故该共性是本轮结果的重要限制，不改格式、不更改得分。','这也说明不能仅凭Pass@4较高就断言PaTR动作序列更丰富：Random的不同动作序列数略高，但不必然更有效。','',
'四、分叉打分的信息量',
'用训练完成的V1 GRPO模型做3×64题纯采样诊断：新PaTR中58/192题活到第5轮，40/192题在第5轮后仍有活分叉，31/40分叉全部平分；在32个含选中和未选中分支的可比叉点中，23/32全部平分。选中后代−未选中成功率均差+1.5625pp，95%CI[-4.6875,+9.375]pp。Random可比32叉，均差0，观测区间[0,0]，不是证明真实差异必为0。',
'用户随后明确撤销采样关卡，要求无论诊断结果如何都训练两组。本轮依该授权执行，并未把诊断关卡报告为通过。']
for label,k in [('新PaTR','patr'),('新Random','tree_random')]:
 r=tree['results'][k]['all'];late=tree['results'][k]['last80'];lines.append(f"{label}正式训练：有活分叉题{r['tasks_with_live_fork']}/{r['sampled_tasks']}；可比叉点{r['comparable_forks']}，其中全平分{r['all_tied_comparable_forks']}；选中后代−未选中均差{100*r['selected_minus_unselected_descendant_success_mean']:+.3f}pp，最后80批{100*late['selected_minus_unselected_descendant_success_mean']:+.3f}pp；score-prune事件{r['event_counts'].get('prune',0)}。")
lines+=['正式训练的这些是观察性统计，受策略变化、剪枝和后代数量不同影响，不能作为随机对照的因果效应。打分有局部预测关联，不保证固定预算下最终测试优于Random。',
'两种新方法不仅扩展选择不同：PaTR还使用低于中位数0.5的过程分剪枝，Random不用。两者都每轮检查连续6次同动作；因此不能称为“只变扩展选择”的严格单因素实验。实际PaTR分数剪枝仅3次，但协议差异仍需披露。','',
'五、训练成本与异常']
for label,k in [('新PaTR','patr'),('新Random','tree_random')]:
 r=tr[k];lines.append(f"{label}: 400批，800次更新；elapsed {r['elapsed_seconds']/3600:.3f}小时；新生成token {r['generated_tokens']:,}；最大记录梯度范数{r['grad_max']:.4f}在第{r['grad_max_step']}批；非有限记录{r['nonfinite_rows']}；参数未变化批次数{len(r['zero_parameter_change_steps'])}；恢复来源步数{r['resumed_from_steps']}；token_limit叶子总数{r['token_limit_episodes']}。")
lines+=['elapsed包含起始验证之后、最后一批更新之前的周期性验证/保存等耗时，不含初始化、最后验证、final test/导出。生成token只计实际新生成，树共享前缀不重复计费；不把每叶子的累计长度相加当算力成本。',
'梯度范数日志是每批两次minibatch裁剪前范数的算术平均，不是单minibatch峰值；max_grad_norm=1执行裁剪。Random第63批39.54的孤立尖峰已记录，普通波动按授权不中断，未据此改超参数。',
'本轮正式训练记录无恢复、无非有限梯度失败；源代码/配置冻结校验通过。推理沿用V1的XFORMERS+enforce_eager。普通状态不打扰、诊断不阻止训练、续租截止延长到10月18日09:00均为用户批准的执行方式；本轮不继承已终止v4训练或其权重。','',
'六、保存与验收',
'两份checkpoint的step均400，optimizer各参数step均800，SHA与训练完成清单一致；最终模型SHA也一致。源码与sequential_frozen_source逐项一致，未改冻结训练文件、未覆盖checkpoint。',
'GitHub轻量归档包含源码、数据任务清单、状态、完整训练指标、所有validation/test轨迹、压缩分叉事件、原始诊断、附加评估、统计脚本与SHA清单。800个原始训练rollout文件留在服务器并登记逐份SHA；权重和optimizer不上传GitHub。权重仍在服务器，租期结束前应另行长期迁移；本报告不声称GitHub含有权重备份。',
'仅seed42，不运行其他seed、GRPO重训或更多消融。完成本轮后关闭训练监控。','',
'最终解释：新树使Random明显追上了PaTR，但不能把单seed的提升唯一归因于某项树改动。当前可报告两者主测试成功率接近，PaTR的Pass@4/格式有效性点估计更好；“PaTR显著优于Random”的核心主张尚未得到本轮支持。']
(out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
with (out/'results.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['method','success_pct','pass4_pct','invalid_action_pct','turns','sampled_tokens','canonical_text_tokens'])
 for k,r in a['results'].items():w.writerow([k,r['success_pct'],r['pass4_pct'],100*r['invalid_action_fraction'],r['turns_mean'],r['sampled_assistant_tokens_mean'],r['canonical_decoded_text_tokens_mean']])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for k,label,color in [('patr','PaTR','#176b8c'),('tree_random','Tree-Random','#ca6434')]:
 rows=[json.loads(s) for s in (root/'runs/formal'/f'{k}_seed42/train_metrics.jsonl').read_text().splitlines()]
 v=tr[k]['validation'];steps=[r['step'] for r in rows]
 ax[0,0].plot([r['step'] for r in v],[r['success_rate_pct'] for r in v],'-o',ms=3,label=label,color=color)
 ax[0,1].plot([r['step'] for r in v],[r['pass_at_4_pct'] for r in v],'-o',ms=3,label=label,color=color)
 ax[1,0].plot(steps,[r['grad_norm'] for r in rows],lw=.8,label=label,color=color,alpha=.8)
 ax[1,1].plot(steps,np.cumsum([r['generated_tokens'] for r in rows])/1e6,lw=1.5,label=label,color=color)
for axis,title,y in zip(ax.flat,['Validation Success (not test)','Validation Pass@4 (not test)','Mean pre-clip minibatch gradient norm','Cumulative new training output tokens'],['Success (%)','Pass@4 (%)','Gradient norm, log scale','Million tokens']):
 axis.set(title=title,xlabel='Training iteration',ylabel=y);axis.grid(alpha=.15);axis.legend(frameon=False)
ax[1,0].set_yscale('log');fig.suptitle('V1 training + revised trees | seed 42 | final checkpoints at iteration 400')
fig.savefig(out/'training_summary.png',dpi=160);fig.savefig(out/'training_summary.svg');plt.close(fig)
print(json.dumps(dict(consistency=matching,report=str(out/'REPORT.txt'))))
