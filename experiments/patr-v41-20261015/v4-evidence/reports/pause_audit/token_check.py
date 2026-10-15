import json,gzip,collections,sys
from pathlib import Path
from transformers import AutoTokenizer
r=Path('/home/vipuser/patr-reproduction-v4');s=r/'snapshots/smoke-20261015-a';sys.path.insert(0,str(s))
from tokenization_v4 import encode_message
from training_contract import filter_and_advantages
tok=AutoTokenizer.from_pretrained('/home/vipuser/patr-reproduction/artifacts/Qwen2.5-0.5B-Instruct',local_files_only=True);a=next((r/'runs/patr-400-20261015-a').glob('attempt-*'));report=[]
for step in [60,68,69,71,72,73,80,85]:
 with gzip.open(a/f'raw_{step-1}.jsonl.gz','rt') as f:bs=[json.loads(l) for l in f]
 groups=[[b for b in bs if b['task_id']==i] for i in range(64)];rows,_=filter_and_advantages([[b['reward'] for b in g] for g in groups]);z=collections.Counter();errors=[]
 for g,j,adv in rows:
  b=groups[g][j];base=len(b['ids'])-b['trace'][-1]['stored_post_prompt_tokens'];prev=0
  for t in b['trace']:
   canonical,_=encode_message(dict(role='assistant',content=t['response']),tok)
   delta=t['stored_post_prompt_tokens']-prev;n=min(delta,len(canonical));actual=b['ids'][base+prev:base+prev+n]
   if actual!=canonical[:n]:errors.append([b['task_id'],t['turn']])
   tag='pos' if adv>0 else 'neg';z[tag+'_turns']+=1
   sampled=t['response_token_ids'];z[tag+'_sampled_eos']+=sampled.count(tok.eos_token_id);z[tag+'_stored_eos']+=actual.count(tok.eos_token_id)
   z[tag+'_length_finished']+=t['finish_reason']=='length'
   if t['finish_reason']=='length' and tok.eos_token_id in actual:z[tag+'_length_with_inserted_eos']+=1
   if canonical!=sampled:
    z['retokenized_turns']+=1
    if canonical==sampled+[198]:z['only_appended_newline']+=1
   prev=t['stored_post_prompt_tokens']
 report.append(dict(step=step,counts=dict(z),assistant_token_mismatch=errors))
(r/'reports/pause_audit/token_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
