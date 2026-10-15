"""Read-only real-Qwen probe; no optimizer, rollout or checkpoint write."""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,json,gzip,time
from pathlib import Path
import torch
from torch.func import functional_call
from transformers import AutoModelForCausalLM
r=Path('/home/vipuser/patr-reproduction-v4');s=r/'snapshots/smoke-20261015-a';sys.path.insert(0,str(s))
from checkpoint_v4 import verified_load
assert json.loads((r/'runs/patr-400-20261015-a/status.json').read_text())['state']=='paused_saved'
a=next((r/'runs/patr-400-20261015-a').glob('attempt-*'));probes=[]
with gzip.open(a/'raw_59.jsonl.gz','rt') as f:
 for l in f:
  b=json.loads(l);t=b['trace'][0];ids=t['response_token_ids']
  if ids and ids[-1]==151645 and len(ids)<=12:
   start=next(i for i,m in enumerate(b['mask']) if m);probes.append((b['ids'][:start]+ids[:-1],t['response']))
  if len(probes)==8:break
model=AutoModelForCausalLM.from_pretrained('/home/vipuser/patr-reproduction/artifacts/Qwen2.5-0.5B-Instruct',torch_dtype=torch.float32,local_files_only=True,attn_implementation='sdpa').cuda().eval()
torch.backends.cuda.matmul.allow_tf32=False
results=[]
def measure(label):
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  params={n:p.to(torch.bfloat16) for n,p in model.model.named_parameters()}
  for i,(tokens,text) in enumerate(probes):
   ids=torch.tensor([tokens],device='cuda');h=functional_call(model.model,params,(),dict(input_ids=ids,use_cache=False)).last_hidden_state[:,-1];p=model.lm_head(h).float().softmax(-1)[0]
   results.append(dict(model=label,probe=i,response=text,im_end=float(p[151645]),endoftext=float(p[151643]),fence=float(p[73594])))
measure('base')
state=verified_load(r/'runs/patr-400-20261015-a/latest.pt');assert state['counters']['accepted_iteration']==87;model.load_state_dict(state['model']);del state
measure('paused87')
x=dict(scope=__doc__,probes=len(probes),results=results,optimizer_updates=0);(r/'reports/pause_audit/ending_probe.json').write_text(json.dumps(x,indent=2));print(json.dumps(x))
