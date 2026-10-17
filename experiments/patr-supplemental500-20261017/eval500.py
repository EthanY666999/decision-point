"""Predeclared fresh 500-task evaluation. No optimizer and no training calls."""
import os,sys,json,time,hashlib,fcntl
from pathlib import Path
import numpy as np
ROOT=Path('/home/vipuser/patr-v1-tree-20261016')
OUT=ROOT/'reports/eval500-20261017'
sys.path.insert(0,str(ROOT))
import train
import patr_core as core

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def cfg():
 c=load(ROOT/'configs/frozenlake_05b.json');c.update(load(ROOT/'artifacts/runtime_override.json'));return c
def paths():return dict(grpo=Path('/home/vipuser/patr-reproduction/runs/formal/grpo_seed42/final_model'),patr=ROOT/'runs/formal/patr_seed42/final_model',tree_random=ROOT/'runs/formal/tree_random_seed42/final_model')
def geometry(t,c):
 e=core.make_env(t,c);rows=[''.join(x.decode() for x in row) for row in e.desc]
 return rows,hashlib.sha256('\n'.join(rows).encode()).hexdigest()
def prepare():
 OUT.mkdir(exist_ok=True)
 assert not (OUT/'protocol.json').exists(),'Never replace a predeclared task set'
 data=load(ROOT/'artifacts/data.json');c=cfg()
 rng=np.random.RandomState(20261017)
 seeds=rng.randint(0,100000,500);sizes=rng.randint(2,10,500);ps=rng.uniform(.6,.85,500)
 tasks=[dict(seed=int(s),size=int(n),p=float(p),uid=f'{s}_{n}_{p}',split='supplemental500') for s,n,p in zip(seeds,sizes,ps)]
 existing={t['uid'] for split in data.values() for t in split}
 assert len({t['uid'] for t in tasks})==500 and not existing.intersection(t['uid'] for t in tasks)
 core.atomic_json(OUT/'tasks.json',tasks)
 maps=[dict(uid=t['uid'],**dict(zip(['desc','geometry_sha256'],geometry(t,c)))) for t in tasks]
 core.atomic_json(OUT/'maps.json',maps)
 old_maps={}
 for split in ['train','validation','test']:
  old_maps[split]={geometry(t,c)[1] for t in data[split]}
 new_hashes=[m['geometry_sha256'] for m in maps]
 core.atomic_json(OUT/'data_audit.json',dict(task_count=500,unique_uids=500,uid_overlap_with_existing=0,unique_geometries=len(set(new_hashes)),additional_draws_repeating_geometry=500-len(set(new_hashes)),geometry_overlap_task_counts={split:sum(h in ss for h in new_hashes) for split,ss in old_maps.items()},geometry_overlap_unique_counts={split:len(set(new_hashes)&ss) for split,ss in old_maps.items()},note='Independent task draws from the original distribution; geometry duplicates are reported, not rejected, to preserve the requested distribution. A new UID does not guarantee unseen map geometry.'))
 model_hashes={k:sha(p/'model.safetensors') for k,p in paths().items()}
 expected=dict(grpo='c70276b0bbce161d257ee98f108b71994f499dd47160079f09a384190ecab287',patr='6dac9a820319f0610356bdeacd56924ccda9d23844b8b7b791e55bd243031c9f',tree_random='72dd22af744b738da44c8de020479e2271fa930c4d160e1bb932baa0455f3ff1')
 assert model_hashes==expected,(model_hashes,expected)
 source=load(ROOT/'reports/sequential_frozen_source.json');assert all(sha(ROOT/n)==v for n,v in source.items())
 main_hashes={str(p.parent/n):sha(p.parent/n) for p in paths().values() for n in ['final_test.json','final_test.jsonl']}
 protocol=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),purpose='Supplemental evaluation only; official original 100-task results remain unchanged and are not pooled with these 500 tasks.',training=False,task_rng='numpy RandomState MT19937',task_seed=20261017,draw_order=['500 map seeds randint(0,100000)','500 sizes randint(2,10)','500 p uniform(.6,.85)'],task_file_sha256=sha(OUT/'tasks.json'),map_file_sha256=sha(OUT/'maps.json'),map_generation_max_steps=10,interaction_max_turns=10,n_tasks=500,samples_per_task=4,evaluation_batches=5,evaluation_tasks_per_batch=100,model_seed=42,evaluation_seed=200042,batch_rng='Reset default_rng(200042) for each fixed 100-task batch, preserving the same base seed in the existing task-UID/lineage/turn sampling protocol.',evaluation_config=c,model_paths={k:str(p) for k,p in paths().items()},model_sha256=model_hashes,frozen_source=source,original_main_test_sha256=main_hashes,statistical_plan='Per-task paired percentile bootstrap, 50000 replicates, analysis RNG42. Separately report new500 results and pairwise differences. Single training seed: no training-seed uncertainty estimated. Report geometry duplication and geometry-cluster sensitivity intervals.',no_selection='No filtering by score/outcome/difficulty, no replacement based on results, no model selection, no extra training. New task IDs disjoint from all original splits.')
 core.atomic_json(OUT/'protocol.json',protocol)
 print(json.dumps(dict(prepared=True,task_sha256=protocol['task_file_sha256'],data_audit=load(OUT/'data_audit.json'))),flush=True)
def evaluate(method):
 protocol=load(OUT/'protocol.json');tasks=load(OUT/'tasks.json');assert sha(OUT/'tasks.json')==protocol['task_file_sha256']
 assert all(sha(ROOT/n)==v for n,v in protocol['frozen_source'].items())
 model=paths()[method];assert sha(model/'model.safetensors')==protocol['model_sha256'][method]
 assert time.time()<load(ROOT/'artifacts/deadline.json')['unix']
 method_out=OUT/method;method_out.mkdir(exist_ok=False)
 c=cfg();c['model']=str(model)
 actor,ref,engine,tokenizer=train.build_models(c,42)
 all_branches=[];totals={'generated_tokens':0,'seconds':0};all_rows=[]
 for batch in range(5):
  assert time.time()<load(ROOT/'artifacts/deadline.json')['unix']
  part=tasks[batch*100:(batch+1)*100]
  core.atomic_json(OUT/'progress.json',dict(method=method,batch=batch+1,completed_tasks=batch*100,phase='sampling',time=time.time()))
  start=time.monotonic()
  branches,stats,_=core.collect(part,engine,tokenizer,c,'grpo',np.random.default_rng(200042),evaluation=True)
  seconds=time.monotonic()-start;metrics=core.evaluation_metrics(branches,part)
  metrics.update(seconds=seconds,generated_tokens=stats['generated_tokens'],batch=batch+1,split='supplemental500')
  records=[]
  for b in branches:
   records.append(dict(task=part[b.task_id],lineage=b.lineage,reward=b.reward,terminal=b.terminal,turns=b.turns,response_tokens=b.ntokens,canonical_text_tokens=sum(len(tokenizer.encode(t['response'],add_special_tokens=False)) for t in b.trace),sampled_assistant_ids=[int(i) for i,m in zip(b.ids,b.mask) if m],trace=b.trace))
   b.task_id+=batch*100
  with (method_out/f'batch{batch+1}.jsonl').open('x') as f:
   for row in records:f.write(json.dumps(row)+'\n')
  core.atomic_json(method_out/f'batch{batch+1}.json',metrics)
  all_rows.extend(records);all_branches.extend(branches)
  totals['seconds']+=seconds;totals['generated_tokens']+=stats['generated_tokens']
  print(json.dumps(dict(method=method,completed_batch=batch+1,completed_tasks=(batch+1)*100,metrics=metrics)),flush=True)
 summary=core.evaluation_metrics(all_branches,tasks);summary.update(totals)
 summary.update(split='supplemental500',canonical_text_tokens_mean=float(np.mean([r['canonical_text_tokens'] for r in all_rows])),model_sha256=protocol['model_sha256'][method],task_sha256=protocol['task_file_sha256'])
 with (method_out/'trajectories.jsonl').open('x') as f:
  for row in all_rows:f.write(json.dumps(row)+'\n')
 core.atomic_json(method_out/'summary.json',summary)
 assert all(sha(Path(p))==h for p,h in protocol['original_main_test_sha256'].items())
 assert sha(model/'model.safetensors')==protocol['model_sha256'][method]
 print(json.dumps(dict(method=method,complete=True,summary=summary)),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 else:evaluate(sys.argv[1])
