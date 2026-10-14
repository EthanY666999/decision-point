"""Diagnose original numerics without writing to or resuming the original run."""
import sys, json, os
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
OLD = Path('/home/vipuser/patr-reproduction')
sys.path.insert(0, str(OLD))
import train as original
import patr_core as c
sys.path.insert(0, str(HERE))
from numerical import verify_qwen_weights, distribution
import torch
import numpy as np
from torch.func import functional_call


def projected_lp(model, ids, attn, mask):
    params = {n: p.to(torch.bfloat16) for n, p in model.model.named_parameters()}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = functional_call(model.model, params, (), {'input_ids': ids, 'attention_mask': attn, 'use_cache': False}).last_hidden_state
        keep = mask[:, 1:].bool()
        logits = model.lm_head(h[:, :-1][keep]).float()
    return -torch.nn.functional.cross_entropy(logits, ids[:, 1:][keep], reduction='none')


def main():
    cfg = json.loads((OLD/'configs/frozenlake_05b.json').read_text())
    cfg.update(json.loads((OLD/'artifacts/runtime_override.json').read_text()))
    cfg.update(max_response_tokens=512, eval_samples=2, micro_batch_size=2, micro_batch_token_limit=5000)
    actor, ref, engine, tok = original.build_models(cfg,42)
    actor.eval()
    data = c.make_datasets()
    result = {'scope': 'Initial model; generated 4 validation tasks x8 branches, measured first 8 returned branches (one task); 512-token diagnostic cap; no optimizer update',
              'weights_initial': verify_qwen_weights(actor, engine)}
    branches, stats, _ = c.collect(data['validation'][:4],engine,tok,cfg,'grpo',np.random.default_rng(20261014),evaluation=False)
    legacy=[]; projected=[]; old=[]; reference=[]; chunk_delta=[]
    for b in branches[:8]:
        ids, attn, mask, _, _, o = original.pack([(b,0,1)],tok.pad_token_id)
        with torch.no_grad():
            a=original.selected_logprobs(actor,ids,attn,mask)
            p=projected_lp(actor,ids,attn,mask)
            r=original.selected_logprobs(ref,ids,attn,mask)
        legacy.append(a.cpu());projected.append(p.cpu());old.append(o.cpu());reference.append(r.cpu())
    a,p,o,r=[torch.cat(x) for x in [legacy,projected,old,reference]]
    result.update(legacy_minus_vllm=distribution(a-o),projected_minus_vllm=distribution(p-o),
                  legacy_minus_reference=distribution(a-r),projected_minus_reference=distribution(p-r),
                  legacy_ratio=distribution((a-o).exp()),projected_ratio=distribution((p-o).exp()),
                  legacy_kl=distribution((r-a).exp()-(r-a)-1),projected_kl=distribution((r-p).exp()-(r-p)-1))
    original.sync_weights(actor,engine)
    result['weights_after_sync']=verify_qwen_weights(actor,engine)
    c.atomic_json(HERE/'reports/numerical_probe_initial.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
