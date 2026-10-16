import os,json,hashlib,importlib.util,sys
from pathlib import Path
import numpy as np
import train
import patr_core as new
root=Path(__file__).resolve().parent
reports=root/'reports/grpo_diagnostic';reports.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('original_v1_core','/home/vipuser/patr-reproduction/patr_core.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
cfg=json.loads((root/'configs/frozenlake_05b.json').read_text());cfg.update(json.loads((root/'artifacts/runtime_override.json').read_text()))
assert (root/'train.py').read_bytes()==Path('/home/vipuser/patr-reproduction/train.py').read_bytes()
for n in ('configs/frozenlake_05b.json','artifacts/data.json','artifacts/runtime_override.json'):
 assert (root/n).read_bytes()==(Path('/home/vipuser/patr-reproduction')/n).read_bytes()
# CPU tests on the actual routing implementation.
from types import SimpleNamespace
import copy
class B:
 def __init__(self,i,pos):
  self.env=SimpleNamespace(s=pos,nrow=2,ncol=3,desc=np.array([list('SFG'),list('FFF')],dtype='S1'));self.initial_distance=1;self.reward=0.;self.actions=[1];self.lineage=str(i);self.turns=5;self.order=(i,);self.terminal=''
 def clone(self,suffix):
  c=copy.deepcopy(self);c.lineage+=suffix;return c
assert new.score_branch(B(0,0),cfg)==1-2/3
bs=[B(i,0) for i in range(4)]
selections=[]
for seed in range(10):
 k,d,e=new.route_patr(copy.deepcopy(bs),cfg,np.random.default_rng(seed));selections.append(tuple(x['lineage'] for x in e if x['kind']=='expand'));assert len(k)==6
assert len(set(selections))>1
k,d,e=new.route_patr([B(0,0),B(1,1),B(2,3),B(3,4)],cfg,np.random.default_rng(42),True);assert not d
assert all(x['score']>=0 for x in e[0]['candidates'])
new.atomic_json(reports/'cpu_gate.json',dict(passed=True,checks=['V1 training/config/data identity','map normalization','random ties','two-way branching','Random no score pruning']))
model=Path('/home/vipuser/patr-reproduction/runs/formal/grpo_seed42/final_model')
status=json.loads((model.parent/'status.json').read_text());assert status['status']=='complete' and status['step']==400 and status['method']=='grpo'
h=hashlib.sha256()
with (model/'model.safetensors').open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
new.atomic_json(reports/'model_provenance.json',dict(model=str(model),sha256=h.hexdigest(),source_status=status,diagnostic_only=True,training_initialization='base model unchanged',decision_policy='diagnostic result never blocks formal training'))
cfg['model']=str(model)
actor,ref,engine,tok=train.build_models(cfg,42)
data=new.make_datasets()['train'];indices=np.random.default_rng(42).choice(len(data),192,replace=False)
results=[]
for batch in range(3):
 tasks=[data[i] for i in indices[batch*64:(batch+1)*64]]
 for label,core,method in [('v1_patr',old,'patr'),('new_patr',new,'patr'),('new_random',new,'tree_random')]:
  bs,stats,events=core.collect(tasks,engine,tok,cfg,method,np.random.default_rng(4200+batch))
  out=reports/f'{label}-{batch}'
  train.write_rollouts(str(out)+'.jsonl.gz',bs,tasks)
  new.atomic_json(str(out)+'.json',dict(stats=stats,events=events,task_uids=[t['uid'] for t in tasks]))
  forks=[e for e in events if e['kind']=='fork_candidates']
  comparable=[e for e in forks if any(not c['selected'] for c in e['candidates'])]
  stats.update(tasks_reaching_turn5=len({b.task_id for b in bs if b.turns>=5}),tasks_with_live_fork_after5=len({e['task_id'] for e in events if e.get('turn')==5 and e['kind'] in ('expand','fork_candidates')}),forks=len(forks),all_tied_forks=sum(len({c['score'] for c in e['candidates']})==1 for e in forks),comparable_forks=len(comparable),all_tied_comparable_forks=sum(len({c['score'] for c in e['candidates']})==1 for e in comparable))
  new.atomic_json(str(out)+'.json',dict(stats=stats,events=events,task_uids=[t['uid'] for t in tasks]))
  differences=[]
  for ev in events:
   if ev['kind']!='fork_candidates':continue
   selected=[];other=[]
   for c in ev['candidates']:
    leaves=[b for b in bs if b.task_id==ev['task_id'] and (b.lineage==c['lineage'] or b.lineage.startswith(c['lineage']+'.'))]
    if leaves:(selected if c['selected'] else other).append(float(np.mean([b.reward>0 for b in leaves])))
   if selected and other:differences.append(float(np.mean(selected)-np.mean(other)))
  results.append(dict(label=label,batch=batch,stats=stats,differences=differences))
  new.atomic_json(reports/'progress.json',dict(completed=len(results),total=9,last=label,batch=batch))
summary={}
for label in ('new_patr','new_random'):
 x=np.array([d for r in results if r['label']==label for d in r['differences']]);rng=np.random.default_rng(42)
 ci=np.quantile(np.mean(rng.choice(x,(10000,len(x)),replace=True),axis=1),[.025,.975]).tolist() if len(x) else [None,None]
 summary[label]=dict(n_forks=len(x),mean_difference=float(x.mean()) if len(x) else None,ci95=ci)
p=summary['new_patr'];r=summary['new_random']
passed=p['n_forks']>=20 and r['n_forks']>=20 and p['ci95'][0]>0 and r['ci95'][0]<=0<=r['ci95'][1]
new.atomic_json(reports/'summary.json',dict(informational_only=True,former_gate_passed=passed,formal_training_independent_of_result=True,comparison=summary,results=results,criterion='predeclared: at least20 paired forks each; PaTR difference95CI lower>0; Random CI contains0',limitation='observational diagnostic; score pruning and unequal descendant counts prevent causal attribution',training_updates=0))
