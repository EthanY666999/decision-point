import sys,json,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import torch
from engine_v4 import build_models,sync_weights,inference_cache_boundary
from training_contract import Counters,EpochSampler
from checkpoint_v4 import restore
cfg=json.loads((ROOT/'config_candidate.json').read_text());out=ROOT/'runs/patr-400-xformers-a'
actor,ref,engine,tok=build_models(cfg,42)
optim=torch.optim.AdamW(actor.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'],weight_decay=cfg['weight_decay'])
data=json.loads((ROOT/'artifacts/data_v4.json').read_text());sampler=EpochSampler(len(data['train']));counts=Counters()
restored=restore(out/'latest.pt',actor,optim,sampler,counts,json.loads((out/'fingerprint.json').read_text()))
from vllm import SamplingParams
samples=[]
for repeat in range(2):
 sync_weights(actor,engine);inference_cache_boundary(engine,'acceptance_before',reset=True)
 result=engine.generate(['Reply with the single word: ready.'],SamplingParams(temperature=1.,max_tokens=16,seed=42+repeat),use_tqdm=False)
 assert result and result[0].outputs[0].token_ids
 samples.append(dict(tokens=result[0].outputs[0].token_ids,finish=result[0].outputs[0].finish_reason))
 inference_cache_boundary(engine,'acceptance_after',reset=True)
report=dict(passed=True,backend=os.environ.get('VLLM_ATTENTION_BACKEND'),enforce_eager=cfg['vllm_enforce_eager'],counts=counts.state_dict(),restore=restored,samples=samples,scope='Fresh XFORMERS startup, full checkpoint restore, two weight sync/cache-reset/generation cycles; optimizer math sources unchanged, prior acceptance reused; not long-run stability proof',time=time.time())
assert report['backend']=='XFORMERS' and counts.accepted_iteration==56
(ROOT/'reports/xformers_runtime_acceptance.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
