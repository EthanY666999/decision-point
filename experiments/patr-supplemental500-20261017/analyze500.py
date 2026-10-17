import sys,json,hashlib,csv,time,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path('/home/vipuser/patr-v1-tree-20261016');sys.path.insert(0,str(ROOT))
import patr_core as core
OUT=ROOT/'reports/eval500-20261017'
load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
protocol=load(OUT/'protocol.json');tasks=load(OUT/'tasks.json');maps=load(OUT/'maps.json');data=load(OUT/'data_audit.json')
assert load(OUT/'controller_status.json')['phase']=='all_three_evaluations_complete'
assert sha(OUT/'tasks.json')==protocol['task_file_sha256']
assert all(sha(Path(p))==h for p,h in protocol['original_main_test_sha256'].items())
assert all(sha(ROOT/p)==h for p,h in protocol['frozen_source'].items())
uids=[t['uid'] for t in tasks];geoms=[m['geometry_sha256'] for m in maps]
assert uids==[m['uid'] for m in maps]
group_ids={h:i for i,h in enumerate(sorted(set(geoms)))};gidx=np.array([group_ids[h] for h in geoms]);ng=len(group_ids);counts=np.bincount(gidx)
parser=core.FrozenLakeAgent(max_steps=10)
results={};arrays={};all_rows={};replay_errors=[]
for method in ['grpo','patr','tree_random']:
 summary=load(OUT/method/'summary.json');rr=[json.loads(s) for s in (OUT/method/'trajectories.jsonl').read_text().splitlines()]
 assert len(rr)==2000 and summary['n_tasks']==500 and summary['n_trajectories']==2000
 groups=defaultdict(list);invalid_by_turn=Counter();invalid_text=Counter();actions=[]
 for r in rr:
  groups[r['task']['uid']].append(r)
  assert len(r['sampled_assistant_ids'])==r['response_tokens']
  env=core.make_env(r['task'],protocol['evaluation_config']);reward=0
  for t in r['trace']:
   action=t.get('action')
   if action is None:continue
   parsed=int(parser._parse_model_response(t['response'])[1]);old=int(env.s)
   obs,rw,done,info=env.step(action);reward+=rw
   if parsed!=action or int(env.s)!=t['state'] or rw!=t['reward']:replay_errors.append([method,r['task']['uid'],r['lineage'],t['turn']])
   actions.append(action)
   if action==0:
    assert old==int(env.s)
    invalid_by_turn[str(t['turn'])]+=1;invalid_text[t['response']]+=1
  assert reward==r['reward']
 assert set(groups)==set(uids) and all(len(v)==4 and len({r['lineage'] for r in v})==4 for v in groups.values())
 s=np.array([np.mean([r['reward']>0 for r in groups[u]]) for u in uids])*100
 p=np.array([max(r['reward']>0 for r in groups[u]) for u in uids])*100
 assert abs(s.mean()-summary['success_rate_pct'])<1e-9 and abs(p.mean()-summary['pass_at_4_pct'])<1e-9
 unique=[len({tuple(t.get('action') for t in r['trace']) for r in groups[u]}) for u in uids]
 arrays[method]=np.stack([s,p],axis=1)
 results[method]=dict(summary,invalid_action_counts_by_turn=dict(invalid_by_turn),most_common_invalid_texts=invalid_text.most_common(10),all_four_same_action_sequence_pct=100*np.mean(np.array(unique)==1),mean_distinct_action_sequences=float(np.mean(unique)))
 core.atomic_json(OUT/method/'per_task.json',[dict(uid=u,geometry_sha256=geoms[i],success_pct=float(s[i]),pass4_pct=float(p[i]),distinct_action_sequences=unique[i]) for i,u in enumerate(uids)])
 all_rows[method]=rr
assert not replay_errors
# One shared resampling matrix per chunk for all paired comparisons and all methods.
methods=['grpo','patr','tree_random'];pairs=[('patr','tree_random'),('patr','grpo'),('tree_random','grpo')]
matrix=np.column_stack([arrays[m] for m in methods]+[arrays[a]-arrays[b] for a,b in pairs])
cluster_sums=np.stack([np.bincount(gidx,weights=matrix[:,j],minlength=ng) for j in range(matrix.shape[1])],axis=1)
rng=np.random.default_rng(42);grng=np.random.default_rng(42);boots=[];gboots=[]
for start in range(0,50000,1000):
 ix=rng.integers(0,500,size=(1000,500));boots.append(matrix[ix].mean(axis=1))
 gi=grng.integers(0,ng,size=(1000,ng));gboots.append(cluster_sums[gi].sum(axis=1)/counts[gi].sum(axis=1)[:,None])
bci=np.quantile(np.concatenate(boots),[.025,.975],axis=0);gci=np.quantile(np.concatenate(gboots),[.025,.975],axis=0)
for mi,m in enumerate(methods):
 for j,key in enumerate(['success','pass4']):
  results[m][key+'_ci95_task_bootstrap']=bci[:,mi*2+j].tolist();results[m][key+'_ci95_geometry_cluster_sensitivity']=gci[:,mi*2+j].tolist()
comparisons={}
for pi,(aa,bb) in enumerate(pairs):
 comparisons[aa+' minus '+bb]={}
 for j,key in enumerate(['success_pp','pass4_pp']):
  d=arrays[aa][:,j]-arrays[bb][:,j];col=6+pi*2+j
  comparisons[aa+' minus '+bb][key]=dict(difference=float(d.mean()),ci95_task_bootstrap=bci[:,col].tolist(),ci95_geometry_cluster_sensitivity=gci[:,col].tolist(),tasks_higher=int(sum(d>0)),tasks_equal=int(sum(d==0)),tasks_lower=int(sum(d<0)))
original={m:load(Path(protocol['model_paths'][m]).parent/'final_test.json') for m in methods}
old_tasks=load(ROOT/'artifacts/data.json')['test']
distribution_comparison={k:dict(n=len(tt),mean_size=float(np.mean([t['size'] for t in tt])),size_counts=dict(Counter(str(t['size']) for t in tt)),mean_p=float(np.mean([t['p'] for t in tt]))) for k,tt in [('official100',old_tasks),('supplemental500',tasks)]}
core.atomic_json(OUT/'distribution_comparison.json',distribution_comparison)
analysis=dict(results=results,comparisons=comparisons,original_100_unchanged=True,original_100_results=original,data_audit=data,protocol_sha256=sha(OUT/'protocol.json'),replay_verified_trajectories=6000,replay_errors=0,bootstrap=dict(replicates=50000,analysis_seed=42,primary='Paired resampling of 500 task UIDs with all four trajectories retained.',sensitivity='Resample 461 unique geometry clusters with replacement; reweight by total task counts in sampled clusters. This is sensitivity to duplicate geometry, not a correction for shared geometry with training.',caveat='Single trained seed42. Intervals quantify task-sampling uncertainty only, not training-seed uncertainty; no multiple-comparison correction.'),training_updates=0)
core.atomic_json(OUT/'analysis.json',analysis)
with (OUT/'results.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['method','Success_pct','Pass4_pct','Success_ci95','Pass4_ci95','invalid_action_pct','mean_turns','sampled_assistant_tokens','canonical_text_tokens'])
 for m in methods:
  r=results[m];w.writerow([m,r['success_rate_pct'],r['pass_at_4_pct'],r['success_ci95_task_bootstrap'],r['pass4_ci95_task_bootstrap'],r['invalid_action_fraction']*100,r['turns'],r['response_length'],r['canonical_text_tokens_mean']])
lines=['新增500题补充评估（只评估、不训练）','日期：2026-10-17；原100题主结果保持不变，未将两组题合并成600题主结果。','',
'协议：沿用原主测试口径。map generation max_steps=10，交互最多10轮；每题4样本；temperature=0.7、top_p=0.8、top_k=20；原V1 GRPO与新树PaTR/Random均使用既有seed42最终模型。没有模型选择、训练或配置调优。',
'任务在评估前固定：numpy RandomState(20261017)，先draw500个map seed [0,100000)，再draw500个size [2,10)，最后draw500个p [.6,.85)。任务清单和地图SHA见protocol.json。为保存进度分为5个固定100题批次；每批沿用default_rng(200042)，逐UID/lineage/turn派生采样种子，三个模型任务与采样协议一致。','',
f"500个新UID与原训练/验证/测试UID零重叠，但仅{data['unique_geometries']}种不同地图几何；{data['additional_draws_repeating_geometry']}次抽样重复本补充集中的几何。{data['geometry_overlap_task_counts']['train']}题的地图几何出现在训练集中，{data['geometry_overlap_task_counts']['test']}题几何出现在原100题test中。保留这些独立随机抽样，不筛掉小地图以改变分布。因此这是500个新任务抽样，不是500个保证从未见过的独特地图。",'',
'原100题与新500题地图尺寸/概率分布实测：'+json.dumps(distribution_comparison,ensure_ascii=False),'两套题集的难度构成会有抽样波动，不能把补充集成功率上升解释为模型训练后进一步提升（没有训练）。','补充500题结果（每个模型2000条轨迹）：','方法 | Success% [95%CI] | Pass@4% [95%CI] | 无效动作% | 平均回合 | sampled tokens | canonical text tokens']
for m in methods:
 r=results[m];lines.append(f"{m} | {r['success_rate_pct']:.2f} {r['success_ci95_task_bootstrap']} | {r['pass_at_4_pct']:.2f} {r['pass4_ci95_task_bootstrap']} | {100*r['invalid_action_fraction']:.2f} | {r['turns']:.4f} | {r['response_length']:.4f} | {r['canonical_text_tokens_mean']:.4f}")
lines+=['','按题配对差值（百分点；50000次percentile bootstrap，分析seed42）：']
for name,cc in comparisons.items():
 for metric,d in cc.items():lines.append(f"{name} {metric}: {d['difference']:+.2f}, 95%CI {d['ci95_task_bootstrap']}；按地图几何成组重采样敏感性区间 {d['ci95_geometry_cluster_sensitivity']}。")
lines+=['','这些区间只描述测试任务抽样不确定性，不涵盖训练随机种子间波动，也不能把区间含0解释为已经证明等效。补充集是在看过原100题结果后按用户要求新增，属于明确标注的后续分析；不追溯替换原主结果。','',
'原100题主结果（只作参照，不更新、不合并）：']
for m in methods:lines.append(f"{m}: Success {original[m]['success_rate_pct']:.2f}%, Pass@4 {original[m]['pass_at_4_pct']:.2f}%")
old=load(ROOT/'reports/closeout-20261017/statistical_analysis.json')['pairs']['patr minus tree_random']['success_pp']['ci95']
new=comparisons['patr minus tree_random']['success_pp']['ci95_task_bootstrap']
lines += [f"PaTR−Random的Success差95%区间宽度：原100题{old[1]-old[0]:.2f}pp，新增500题{new[1]-new[0]:.2f}pp。这是两套不同抽样集的实测比较，不能保证所有不确定性随样本数消失。",'',
'长度定义与评分：sampled是实际采样assistant IDs数，包括输出终止token，不含prompt/环境；canonical text是每回合解码响应重新分词、不加特殊token。无效动作按原严格解析器计分，无修复、无重评分；环境允许无效后继续交互。']
for m in methods:lines.append(f"{m}: 无效动作按回合计数{results[m]['invalid_action_counts_by_turn']}；四条动作序列完全相同题占{results[m]['all_four_same_action_sequence_pct']:.1f}%。")
lines+=['','验收：6000条轨迹动作解析、环境状态转移和奖励已逐条重放一致；采样token计数与保存的IDs长度一致；每题4条不同lineage；模型权重、冻结源码及原主test文件SHA均保持不变。完整采样/审计脚本、固定任务/地图、原始响应、分批指标、统计和来源清单一同归档。','训练更新次数：0。']
(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(results={m:{k:results[m][k] for k in ['success_rate_pct','pass_at_4_pct','invalid_action_fraction']} for m in methods},comparisons=comparisons,replay_verified=6000)))
