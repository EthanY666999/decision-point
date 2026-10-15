"""Local logit derivatives on one historical full group using paused87 weights.
This is a counterfactual replay, not reconstruction of step73 parameter gradients.
No optimizer update, generation, or checkpoint write.
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,json,gzip,time,math
from pathlib import Path
import torch
from torch.func import functional_call
from transformers import AutoModelForCausalLM
r=Path('/home/vipuser/patr-reproduction-v4');s=r/'snapshots/smoke-20261015-a';sys.path.insert(0,str(s))
from training_contract import filter_and_advantages
from checkpoint_v4 import verified_load
assert json.loads((r/'runs/patr-400-20261015-a/status.json').read_text())['state']=='paused_saved'
a=next((r/'runs/patr-400-20261015-a').glob('attempt-*'))
with gzip.open(a/'raw_72.jsonl.gz','rt') as f:bs=[json.loads(l) for l in f]
groups=[[b for b in bs if b['task_id']==i] for i in range(64)];rows,_=filter_and_advantages([[b['reward'] for b in g] for g in groups]);eligible={g for g,j,adv in rows};g=max(eligible,key=lambda g:sum(sum(b['mask']) for b in groups[g] if b['reward']==0));selected=[(groups[g][j],adv) for gg,j,adv in rows if gg==g]
from types import SimpleNamespace
from update_v4 import update
from training_contract import Counters
from numerical import projected_checkpoint
from engine_v4 import selected_logprobs
model=AutoModelForCausalLM.from_pretrained('/home/vipuser/patr-reproduction/artifacts/Qwen2.5-0.5B-Instruct',torch_dtype=torch.float32,local_files_only=True,attn_implementation='sdpa').cuda().eval()
state=verified_load(r/'runs/patr-400-20261015-a/latest.pt');assert state['counters']['accepted_iteration']==87;model.load_state_dict(state['model'])
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=.01);del state  # Diagnostic control: reset Adam state only
model._project_forward_bf16=True;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.model._gradient_checkpointing_func=projected_checkpoint
reference=AutoModelForCausalLM.from_pretrained('/home/vipuser/patr-reproduction/artifacts/Qwen2.5-0.5B-Instruct',torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').cuda().eval()
for p in reference.parameters():p.requires_grad_(False)
branches=[SimpleNamespace(ids=b['ids'],mask=b['mask'],ntokens=sum(b['mask']),reward=b['reward'],task_id=0) for b,adv in selected]
b=next(b for b,adv in selected if b['reward']==1);start=next(i for i,m in enumerate(b['mask']) if m);t=b['trace'][0];probe=b['ids'][:start]+t['response_token_ids'][:-1]
def measure():
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  params={n:p.to(torch.bfloat16) for n,p in model.model.named_parameters()};h=functional_call(model.model,params,(),dict(input_ids=torch.tensor([probe],device='cuda'),use_cache=False)).last_hidden_state[:,-1];p=model.lm_head(h).float().softmax(-1)[0]
  return {str(k):float(p[k]) for k in [151643,151645,73594]}
before=measure();tick=time.time();result=update(model,reference,optimizer,branches,SimpleNamespace(pad_token_id=151645),Counters(),1);after=measure()
x=dict(scope='Single whole historical group from step73 replayed on paused87 model with reset Adam state; diagnostic only, no saved model, no formal resume, no Success evaluation',group=g,before=before,after=after,update=result,seconds=time.time()-tick)
(r/'reports/pause_audit/reset_adam_probe.json').write_text(json.dumps(x,indent=2));print(json.dumps(x))
