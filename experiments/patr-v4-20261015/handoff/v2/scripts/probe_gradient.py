"""Real GPU regression: projected forward, checkpointed gradients, repeatability."""
import sys,json,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import numpy as np
import train as t
import patr_core as c
from protocol_v2 import profile_config
from numerical import distribution

cfg=json.loads((ROOT/'configs/frozenlake_05b.json').read_text())
cfg.update(json.loads((ROOT/'artifacts/runtime_override.json').read_text()))
cfg.update(profile_config('behavior_variant_v2'))
cfg.update(project_forward_bf16=True,recompute_old_logprobs=True,kl_estimator='categorical',
           rollout_is_max=2.0,diagnostic_gradient_split=True,mini_batch_tasks=1,
           micro_batch_size=2,micro_batch_token_limit=6000)
actor,ref,engine,tok=t.build_models(cfg,42)
path=Path('/home/vipuser/patr-reproduction/runs/formal/patr_seed42/rollouts_0200.jsonl.gz')
bs=[]
with gzip.open(path,'rt') as f:
    for line in f:
        r=json.loads(line)
        if len(r['ids'])>2500:continue
        bs.append(c.Branch(0,None,None,r['ids'],r['mask'],r['old_logprobs'],reward=float(len(bs)%2),trace=r['trace']))
        if len(bs)==2:break
assert len(bs)==2
rows=[(b,float(i*2-1),1/(2*b.ntokens)) for i,b in enumerate(bs)]
ids,attn,mask,adv,w,old=t.pack(rows,tok.pad_token_id)
actor.train()
repeats=[];grads=[]
for _ in range(2):
    actor.zero_grad(set_to_none=True)
    lp,kl=t.selected_exact_kl(actor,ref,ids,attn,mask)
    loss=((-lp*adv+cfg['kl_coefficient']*kl)*w).sum()
    loss.backward()
    g=actor.model.layers[-1].mlp.down_proj.weight.grad.detach().clone()
    early_grad=actor.model.layers[0].mlp.down_proj.weight.grad
    embed_grad=actor.model.embed_tokens.weight.grad
    assert early_grad is not None and torch.isfinite(early_grad).all() and early_grad.abs().sum()>0
    assert embed_grad is not None and torch.isfinite(embed_grad).all() and embed_grad.abs().sum()>0
    repeats.append(dict(loss=loss.item(),kl=distribution(kl),grad_norm=g.norm().item(),first_layer_grad_norm=early_grad.norm().item(),embedding_grad_norm=embed_grad.norm().item()))
    grads.append(g)
assert torch.isfinite(grads[0]).all() and grads[0].abs().sum()>0
torch.testing.assert_close(grads[0],grads[1],rtol=1e-5,atol=1e-6)
assert all(abs(r['kl']['max'])<1e-5 for r in repeats)
actor.zero_grad(set_to_none=True)
opt=torch.optim.AdamW(actor.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
result=t.update(actor,ref,opt,bs,cfg,'patr',tok,np.random.default_rng(42),1)
assert result['max_parameter_change']>0 and np.isfinite(result['loss'])
t.sync_weights(actor,engine)
c.atomic_json(ROOT/'reports/gradient_probe.json',dict(repeated_forward=repeats,gradient_max_abs_difference=(grads[0]-grads[1]).abs().max().item(),update=result,passed=True))
print('PROJECTED_CHECKPOINT_GRADIENT_AND_REAL_UPDATE_PASS',flush=True)
