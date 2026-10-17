import sys,json,time,hashlib
from pathlib import Path
ROOT=Path('/home/vipuser/patr-v1-tree-20261016')
sys.path.insert(0,str(ROOT))
import train
import patr_core as core
import numpy as np
OUT=ROOT/'reports/closeout-20261017'
OUT.mkdir(exist_ok=True)
cfg=json.loads((ROOT/'configs/frozenlake_05b.json').read_text())
cfg.update(json.loads((ROOT/'artifacts/runtime_override.json').read_text()))
tasks=json.loads((ROOT/'artifacts/data.json').read_text())['test']
def env8(task,cfg):
 e=core.FrozenLakeEnv(size=task['size'],p=task['p'],seed=task['seed'],is_slippery=cfg['is_slippery'],max_steps=8)
 e.reset();return e
maps=[]
for t in tasks:
 a=core.make_env(t,cfg);b=env8(t,cfg)
 maps.append(dict(uid=t['uid'],changed=not np.array_equal(a.desc,b.desc),map10=[''.join(x.decode() for x in r) for r in a.desc],map8=[''.join(x.decode() for x in r) for r in b.desc],distance10=core.distance(a),distance8=core.distance(b)))
core.atomic_json(OUT/'map_comparison.json',dict(n_tasks=len(tasks),changed=sum(m['changed'] for m in maps),interaction_horizon=cfg['max_turns'],note='max_steps changes map generation DFS validity parameter only; interaction horizon remains 10',maps=maps))
if sys.argv[1]=='maps':raise SystemExit
method=sys.argv[1]
model=(Path('/home/vipuser/patr-reproduction/runs/formal/grpo_seed42/final_model') if method=='grpo' else ROOT/'runs/formal'/f'{method}_seed42/final_model')
assert not (OUT/f'{method}_map8.json').exists()
assert time.time()<1792284300
cfg['model']=str(model)
core.make_env=env8
actor,ref,engine,tokenizer=train.build_models(cfg,42)
start=time.time()
branches,stats,_=core.collect(tasks,engine,tokenizer,cfg,'grpo',np.random.default_rng(200042),evaluation=True)
metrics=core.evaluation_metrics(branches,tasks)
metrics.update(seconds=time.time()-start,generated_tokens=stats['generated_tokens'],split='test_map_generation_max_steps8_interaction_horizon10',model=str(model),model_sha256=hashlib.sha256((model/'model.safetensors').read_bytes()).hexdigest(),evaluation_seed=200042,config=cfg)
rows=[]
for b in branches:
 sampled=[int(i) for i,m in zip(b.ids,b.mask) if m]
 canonical=sum(len(tokenizer.encode(t['response'],add_special_tokens=False)) for t in b.trace)
 rows.append(dict(task=tasks[b.task_id],lineage=b.lineage,reward=b.reward,terminal=b.terminal,turns=b.turns,response_tokens=b.ntokens,canonical_text_tokens=canonical,sampled_assistant_ids=sampled,trace=b.trace))
with (OUT/f'{method}_map8.jsonl').open('x') as f:
 for row in rows:f.write(json.dumps(row)+'\n')
metrics['canonical_text_tokens_mean']=float(np.mean([r['canonical_text_tokens'] for r in rows]))
core.atomic_json(OUT/f'{method}_map8.json',metrics)
print(json.dumps(metrics),flush=True)
