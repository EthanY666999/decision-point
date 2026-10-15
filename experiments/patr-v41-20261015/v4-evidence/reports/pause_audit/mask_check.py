import gzip,json
from pathlib import Path
r=Path('/home/vipuser/patr-reproduction-v4');a=next((r/'runs/patr-400-20261015-a').glob('attempt-*'));count=0;errors=[]
for step in [60,68,69,70,71,72,73,76,80,85]:
 with gzip.open(a/f'raw_{step-1}.jsonl.gz','rt') as f:
  for line in f:
   b=json.loads(line);trace=b['trace'];stored=trace[-1]['stored_post_prompt_tokens'];start=len(b['mask'])-stored
   expect=[0]*start;prev=0
   for t in trace:
    delta=t['stored_post_prompt_tokens']-prev;n=min(delta,t['canonical_assistant_tokens']);expect.extend([1]*n+[0]*(delta-n));prev=t['stored_post_prompt_tokens']
   if expect!=b['mask'] or len(b['ids'])!=len(expect):errors.append([step,b['task_id'],b['lineage']])
   count+=1
x=dict(trajectories_checked=count,mask_reconstruction_errors=errors,scope='Reconstruct masks from retained per-turn stored lengths and canonical assistant lengths; checks environment/prompt masking and truncated placement, not tokenizer semantics or gradient equivalence')
(r/'reports/pause_audit/mask_check.json').write_text(json.dumps(x,indent=2));print(json.dumps(x))
