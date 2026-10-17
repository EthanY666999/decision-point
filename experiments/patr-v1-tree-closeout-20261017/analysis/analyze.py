import sys,json,hashlib,math,re,time
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path('/home/vipuser/patr-v1-tree-20261016');sys.path.insert(0,str(ROOT))
import patr_core as core
from transformers import AutoTokenizer
OUT=ROOT/'reports/closeout-20261017';OUT.mkdir(exist_ok=True)
BASE=Path('/home/vipuser/patr-reproduction')
cfg=json.loads((ROOT/'configs/frozenlake_05b.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(str(ROOT/cfg['model']),local_files_only=True)
tasks=json.loads((ROOT/'artifacts/data.json').read_text())['test'];uids=[t['uid'] for t in tasks]
parser=core.FrozenLakeAgent(max_steps=10)
indices=np.random.default_rng(42).integers(0,100,size=(50000,100))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def ci(a):return [float(x) for x in np.quantile(a[indices].mean(1),[.025,.975])]
def inspect(name,p,map8=False):
 rr=rows(p);assert len(rr)==400
 grouped=defaultdict(list);can=[];invalids=Counter();replay_errors=[];nt=0;state_unchanged_invalid=0
 for r in rr:
  grouped[r['task']['uid']].append(r)
  e=core.FrozenLakeEnv(size=r['task']['size'],p=r['task']['p'],seed=r['task']['seed'],is_slippery=False,max_steps=8 if map8 else 10);e.reset()
  reward=0
  can.append(sum(len(tokenizer.encode(t['response'],add_special_tokens=False)) for t in r['trace']))
  for t in r['trace']:
   a=t.get('action')
   if a is None:continue
   nt+=1
   parsed=int(parser._parse_model_response(t['response'])[1])
   if parsed!=a:replay_errors.append(dict(uid=r['task']['uid'],lineage=r['lineage'],kind='parser'))
   old=int(e.s);obs,rw,done,info=e.step(a);reward+=rw
   if int(e.s)!=t['state'] or rw!=t['reward']:replay_errors.append(dict(uid=r['task']['uid'],lineage=r['lineage'],kind='state_reward'))
   if a==0:
    state_unchanged_invalid+=int(old==int(e.s))
    if not re.findall(r'```(.*?)```',t['response'],re.S):invalids['missing_complete_triple_backtick_fence']+=1
    else:invalids['unrecognized_fenced_content']+=1
  if reward!=r['reward']:replay_errors.append(dict(uid=r['task']['uid'],lineage=r['lineage'],kind='trajectory_reward'))
 assert set(grouped)==set(uids) and all(len(v)==4 for v in grouped.values())
 assert not replay_errors,replay_errors[:3]
 success=np.array([np.mean([r['reward']>0 for r in grouped[u]]) for u in uids])*100
 passed=np.array([max(r['reward']>0 for r in grouped[u]) for u in uids])*100
 unique=[len({tuple(t.get('action') for t in r['trace']) for r in grouped[u]}) for u in uids]
 per=[dict(uid=u,success_pct=float(success[i]),pass4_pct=float(passed[i]),distinct_action_sequences=unique[i]) for i,u in enumerate(uids)]
 core.atomic_json(OUT/f'{name}_per_task.json',per)
 result=dict(success_pct=float(success.mean()),success_ci95_task_bootstrap=ci(success),pass4_pct=float(passed.mean()),pass4_ci95_task_bootstrap=ci(passed),sampled_assistant_tokens_mean=float(np.mean([r['response_tokens'] for r in rr])),canonical_decoded_text_tokens_mean=float(np.mean(can)),turns_mean=float(np.mean([r['turns'] for r in rr])),invalid_action_fraction=sum(invalids.values())/nt,invalid_categories=dict(invalids),invalid_state_unchanged_count=state_unchanged_invalid,total_actions=nt,all_four_same_action_sequence_pct=100*sum(n==1 for n in unique)/100,mean_distinct_action_sequences=float(np.mean(unique)),terminal_counts=dict(Counter(r['terminal'] for r in rr)),replay=dict(trajectories=400,parser_state_reward_errors=0),source_sha256=sha(p))
 return result,(success,passed)
all_results={};arrays={}
for name,p in [('grpo',BASE/'runs/formal/grpo_seed42/final_test.jsonl'),('patr',ROOT/'runs/formal/patr_seed42/final_test.jsonl'),('tree_random',ROOT/'runs/formal/tree_random_seed42/final_test.jsonl'),('old_patr',BASE/'runs/formal/patr_seed42/final_test.jsonl'),('old_tree_random',BASE/'runs/formal/tree_random_seed42/final_test.jsonl')]:
 all_results[name],arrays[name]=inspect(name,p)
for method in ['grpo','patr','tree_random']:
 p=OUT/f'{method}_map8.jsonl'
 if p.exists():all_results[method+'_map8'],arrays[method+'_map8']=inspect(method+'_map8',p,True)
pairs={}
for a,b in [('patr','tree_random'),('patr','grpo'),('tree_random','grpo'),('patr','old_patr'),('tree_random','old_tree_random'),('patr_map8','tree_random_map8'),('patr_map8','grpo_map8'),('tree_random_map8','grpo_map8')]:
 if a not in arrays or b not in arrays:continue
 pairs[a+' minus '+b]={}
 for i,metric in enumerate(['success_pp','pass4_pp']):
  d=arrays[a][i]-arrays[b][i]
  pairs[a+' minus '+b][metric]=dict(difference=float(d.mean()),ci95=ci(d),tasks_higher=int(sum(d>0)),tasks_equal=int(sum(d==0)),tasks_lower=int(sum(d<0)))
core.atomic_json(OUT/'statistical_analysis.json',dict(method='Paired bootstrap resampling 100 task UIDs, preserving all four samples per task; 50000 replicates, analysis seed42, percentile95. Single training seed; CIs do not include training-seed variability. Task bootstrap assumes independent task IDs; duplicate map geometry may reduce effective sample size.',length_definitions=dict(sampled='Sum of original sampled assistant token IDs including generated terminal tokens, excludes prompt/environment. Main test stores count, supplementary test also stores exact IDs.',canonical='Sum of tokenizer.encode(decoded assistant response, add_special_tokens=False) per turn. Excludes omitted special/EOS tokens. This is text-retokenization, not training canonicalization; neither number establishes equivalence to paper RLen.'),results=all_results,pairs=pairs))
training={}
for method in ['patr','tree_random']:
 run=ROOT/'runs/formal'/f'{method}_seed42';rr=rows(run/'train_metrics.jsonl');assert [r['step'] for r in rr]==list(range(1,401))
 vals=[r for r in rr if not all(math.isfinite(r[k]) for k in ['grad_norm','kl','loss','max_parameter_change'])];assert not vals
 training[method]=dict(steps=len(rr),elapsed_seconds=rr[-1]['elapsed_seconds'],generated_tokens=sum(r['generated_tokens'] for r in rr),input_tokens=sum(r['input_tokens'] for r in rr),episodes=sum(r['episodes'] for r in rr),training_invalid_action_fraction=sum(r['invalid_actions'] for r in rr)/sum(r['executed_actions'] for r in rr),token_limit_episodes=sum(r['terminal_counts'].get('token_limit',0) for r in rr),grad_median=float(np.median([r['grad_norm'] for r in rr])),grad_max=max(r['grad_norm'] for r in rr),grad_max_step=max(rr,key=lambda r:r['grad_norm'])['step'],gradient_metric_definition='Arithmetic mean of two optimizer minibatch pre-clip norms; not the individual minibatch maximum.',nonfinite_rows=len(vals),zero_parameter_change_steps=[r['step'] for r in rr if r['max_parameter_change']==0],resumed_from_steps=sorted(set(r['resumed_from_step'] for r in rr)),last=rr[-1],validation=[dict(step=int(p.stem.split('_')[1]),**load(p)) for p in sorted(run.glob('validation_*.json'))])
core.atomic_json(OUT/'training_analysis.json',training)
if '--integrity' in sys.argv:
 import torch,gc
 identity=load(ROOT/'reports/sequential_frozen_source.json');assert all(sha(ROOT/n)==h for n,h in identity.items())
 integrity=dict(source_matches_frozen=True,source=identity,models={})
 for method in ['patr','tree_random']:
  run=ROOT/'runs/formal'/f'{method}_seed42';old=load(ROOT/'reports'/f'{method}_completion.json')
  cp_sha=sha(run/'checkpoint.pt');model_sha=sha(run/'final_model/model.safetensors')
  assert cp_sha==old['checkpoint_sha256'] and model_sha==old['final_model_sha256']
  cp=torch.load(run/'checkpoint.pt',map_location='cpu',mmap=True,weights_only=False)
  updates=sorted(set(int(v['step']) for v in cp['optimizer']['state'].values()))
  assert cp['step']==400 and updates==[800]
  integrity['models'][method]=dict(checkpoint_sha256=cp_sha,final_model_sha256=model_sha,checkpoint_step=cp['step'],optimizer_update_counts=updates,status=load(run/'status.json'))
  del cp;gc.collect()
 core.atomic_json(OUT/'integrity.json',integrity)
print(json.dumps(dict(results={k:{n:v[n] for n in ['success_pct','pass4_pct','sampled_assistant_tokens_mean','canonical_decoded_text_tokens_mean']} for k,v in all_results.items()},pairs=pairs,training={k:{n:v[n] for n in ['elapsed_seconds','generated_tokens','grad_max','grad_max_step']} for k,v in training.items()})))
