"""Descriptive CPU diagnostic; sampled-token minimum probability is not full entropy."""
import gzip,json,math,sys,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));import patr_core as core
ROOT=Path(__file__).resolve().parents[1]
NEXT=Path('/home/vipuser/golden-next-20261031')
SOURCES={
 'b2_patr_k2':(NEXT/'b2/code/runs/formal/patr_seed42',[1,2,3,4,5,10,20,60,100,150]),
 'seed43_patr_k5':(NEXT/'seed43_patr/code/runs/formal/patr_seed43',[1,20,60,150,250,400]),
 'seed43_random_k5':(NEXT/'seed43_tree_random/code/runs/formal/tree_random_seed43',[1,20,60,150,250,400]),
 'sft_grpo400':(Path('/home/vipuser/golden-sft-patr-20261031/code/runs/formal/grpo_seed42'),[1,20,60,150,250,400])}
cfg=json.loads((ROOT/'configs/cert_g2_uct.json').read_text())
report={'purpose':'descriptive only; no gate or tuning','proxy':'minimum probability among sampled response tokens including formatting; NOT H_cons or full action entropy','weighting':'leaf-weighted; shared prefixes repeated; separate prefix-unique view','sources':{},'missing':[]}
for label,(root,steps) in SOURCES.items():
 for step in steps:
  path=root/f'rollouts_{step:04d}.jsonl.gz'
  if not path.exists():report['missing'].append(str(path));continue
  groups=defaultdict(list);unique=defaultdict(list);seen=set();mismatch=0;count=0
  for line in gzip.open(path,'rt'):
   row=json.loads(line);task=row['task'];env=core.make_env(task,cfg);state=int(env.s)
   spans=[];start=None
   for i,m in enumerate(row['mask']+[0]):
    if m and start is None:start=i
    if not m and start is not None:spans.append((start,i));start=None
   trace=row['trace']
   if len(spans)!=len(trace):mismatch+=1;continue
   for turn,((lo,hi),tr) in enumerate(zip(spans,trace),1):
    probs=[math.exp(float(x)) for x in row['old_logprobs'][lo:hi]]
    assert probs and all(math.isfinite(p) and 0<=p<=1+1e-5 for p in probs)
    pmin=min(probs);r,c=divmod(state,env.ncol)
    wall=r in (0,env.nrow-1) or c in (0,env.ncol-1)
    hole=any(0<=r+dr<env.nrow and 0<=c+dc<env.ncol and env.desc[r+dr,c+dc]==b'H' for dr,dc in ((0,-1),(1,0),(0,1),(-1,0)))
    keys=[f'turn_{turn}', 'all', 'wall' if wall else 'nonwall','hole_adjacent' if hole else 'not_hole_adjacent']
    if not wall and not hole:keys.append('ordinary')
    action=tr.get('action');fell=False
    if action is not None:
     _,_,done,_=env.step(action);fell=done and not env.success();state=int(env.s)
    if fell:keys.extend(['hole_decision',f'hole_decision_turn_{turn}'])
    if row['terminal']=='success':keys.append(f'success_leaf_turn_{turn}')
    key=(task['uid'],tuple(row['ids'][:lo]));fresh=key not in seen;seen.add(key)
    for k in keys:
     groups[k].append(pmin)
     if fresh:unique[k].append(pmin)
    count+=1
  def summarize(gs):
   return {k:dict(n=len(v),mean_min_probability=float(np.mean(v)),fraction_p_below_06=float(np.mean(np.array(v)<.6)),fraction_p_above_099=float(np.mean(np.array(v)>.99))) for k,v in gs.items()}
  report['sources'][f'{label}_step{step}']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),response_nodes=count,unmatched_trace_rows=mismatch,leaf_weighted=summarize(groups),prefix_unique=summarize(unique))
out=ROOT/'runs/g0';out.mkdir(parents=True,exist_ok=True);core.atomic_json(out/'summary.json',report)
print(json.dumps({'files':len(report['sources']),'missing':report['missing'],'output':str(out/'summary.json')}))
