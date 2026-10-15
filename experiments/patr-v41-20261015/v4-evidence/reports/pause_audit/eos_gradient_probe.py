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
model=AutoModelForCausalLM.from_pretrained('/home/vipuser/patr-reproduction/artifacts/Qwen2.5-0.5B-Instruct',torch_dtype=torch.float32,local_files_only=True,attn_implementation='sdpa').cuda().eval();state=verified_load(r/'runs/patr-400-20261015-a/latest.pt');assert state['counters']['accepted_iteration']==87;model.load_state_dict(state['model']);del state
torch.backends.cuda.matmul.allow_tf32=False;reports=[]
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 params={n:p.to(torch.bfloat16) for n,p in model.model.named_parameters()}
 for b,adv in selected:
  ids=torch.tensor([b['ids']],device='cuda');mask=torch.tensor(b['mask'][1:],device='cuda').bool();labels=ids[0,1:];hidden=functional_call(model.model,params,(),dict(input_ids=ids,use_cache=False)).last_hidden_state[0,:-1];sums={151643:0.,151645:0.};target_counts={k:0 for k in sums};probs={k:0. for k in sums}
  for lo in range(0,len(labels),512):
   logits=model.lm_head(hidden[lo:lo+512]).float();logz=logits.logsumexp(-1);m=mask[lo:lo+512];y=labels[lo:lo+512]
   for k in sums:
    p=(logits[:,k]-logz).exp();target=(y==k).float();d=adv*(p-target)/(256*10240);sums[k]+=float(d[m].sum());target_counts[k]+=int(target[m].sum());probs[k]+=float(p[m].sum())
  reports.append(dict(lineage=b['lineage'],reward=b['reward'],advantage=adv,assistant_tokens=int(mask.sum()),eos_logit_gradient_sum=sums,eos_target_counts=target_counts,eos_probability_sum=probs));del hidden
result=dict(scope=__doc__,task_id=g,step_of_trajectories=73,model_step=87,derivative_definition='dL/dlogit = advantage*(p-onehot)/(256*10240), at freshly recomputed ratio1; positive means local gradient descent lowers that logit; not actual shared-parameter update',rows=reports,optimizer_updates=0)
assert all(math.isfinite(v) for x in reports for v in x['eos_logit_gradient_sum'].values());(r/'reports/pause_audit/eos_gradient_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
