"""Paired validation of frozen original endpoints; no training, no test access."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from safetensors.torch import load_file
import train as t
from patr_core import atomic_json,make_datasets
from protocol_v2 import profile_config
from numerical import verify_qwen_weights

cfg=json.loads((ROOT/'configs/frozenlake_05b.json').read_text())
cfg.update(json.loads((ROOT/'artifacts/runtime_override.json').read_text()))
cfg.update(dataset_path='artifacts/data_v2.json',micro_batch_size=2,micro_batch_token_limit=6000)
actor,ref,engine,tok=t.build_models(cfg,42)
tasks=make_datasets(cfg)['validation']
assert len(tasks)==100 and all(x['split']=='validation' for x in tasks)
out=ROOT/'runs/frozen_prompt_validation'
out.mkdir(parents=True,exist_ok=True)
for method in ['patr','grpo','tree_random']:
    path=Path('/home/vipuser/patr-reproduction/runs/formal')/(method+'_seed42')/'final_model/model.safetensors'
    state=load_file(str(path),device='cpu')
    result=actor.load_state_dict(state,strict=False)
    assert set(result.missing_keys)<= {'lm_head.weight'} and not result.unexpected_keys
    del state
    t.sync_weights(actor,engine)
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    metadata=dict(method=method,source_model=str(path),source_sha256=digest.hexdigest(),optimizer_updates=0,
                  dataset_path=cfg['dataset_path'],split='validation',seed=100042,
                  note='Same frozen endpoint and task seeds; prompt intervention only. No inference of retraining gains.',
                  sync=verify_qwen_weights(actor,engine))
    atomic_json(out/(method+'_source.json'),metadata)
    for profile in ['legacy','corrected_v2']:
        run_cfg={**cfg,**profile_config(profile)}
        # Pruning and training reward settings are inactive in evaluation.
        metrics=t.evaluate(engine,tok,run_cfg,tasks,out/(method+'_'+profile+'.json'),100042,'validation')
        print(json.dumps(dict(method=method,profile=profile,metrics=metrics)),flush=True)
if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
