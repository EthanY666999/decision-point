"""A1/A2 read-only evaluations, using the frozen controls implementation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path('/home/vipuser/golden-controls-20261031/code')
OLD = Path('/home/vipuser/golden-sft-patr-20261031/code')
EXT = Path('/home/vipuser/golden-patr-extended800-20261031/code')
MODELS = {'sft_grpo400': OLD/'runs/formal/grpo_seed42/final_model',
          'sft_patr400': OLD/'runs/formal/patr_seed42/final_model',
          'sft_patr500': EXT/'runs/formal/patr_seed42/final_model',
          'sft_tree_random400': FROZEN/'runs/formal/tree_random_seed42/final_model'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',choices=MODELS,required=True)
    p.add_argument('--protocol',choices=['map8','cross_gpu_main100'],required=True)
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args()
    sys.path.insert(0,str(FROZEN))
    import patr_core as core
    cfg=json.loads((FROZEN/'configs/tree_random400.json').read_text())
    cfg['model']=str(MODELS[a.label])
    tasks=json.loads((FROZEN/'artifacts/data.json').read_text())['test']
    source_maps=json.loads((OLD.parent/'results/main100/analysis/map_comparison.json').read_text())['maps']
    reference={r['uid']:r for r in source_maps}
    def env8(task,cfg):
        env=core.FrozenLakeEnv(size=task['size'],p=task['p'],seed=task['seed'],
                              is_slippery=cfg['is_slippery'],max_steps=8)
        env.reset();return env
    maker=env8 if a.protocol=='map8' else core.make_env
    maps=[]
    for t in tasks:
        env=maker(t,cfg)
        board=[''.join(c.decode() for c in row) for row in env.desc]
        assert board==reference[t['uid']]['map8' if a.protocol=='map8' else 'map10']
        maps.append({'uid':t['uid'],'map':board})
    assert len(tasks)==100 and cfg['max_turns']==10
    assert sum(r['changed'] for r in source_maps)==8
    if a.audit_only:
        print(json.dumps({'maps_match_archived_protocol':100,'horizon':10,'protocol':a.protocol}));return
    assert os.environ.get('NEXT_GPU_LEASE_FD'), 'Run under serialized GPU queue'
    os.fstat(int(os.environ['NEXT_GPU_LEASE_FD']))
    out=ROOT/'runs/evaluations'/a.protocol/a.label
    out.mkdir(parents=True,exist_ok=False)
    core.atomic_json(out/'identity.json',dict(label=a.label,protocol=a.protocol,config=cfg,
        model_sha256=sha(MODELS[a.label]/'model.safetensors'),model_seed=42,evaluation_seed=200042,
        task_file_sha256=sha(FROZEN/'artifacts/data.json'),maps=maps,
        frozen_sources={f:sha(FROZEN/f) for f in ['train.py','patr_core.py']},
        hardware='RTX4090 48GiB',note='Map8 matches archived rLLM parameter ablation, not verified official paper data. Cross-GPU repeat also includes fresh-process/inference-order variability; not a pure hardware causal estimate.'))
    if a.protocol=='map8':core.make_env=env8
    from train import build_models,evaluate
    start=time.monotonic()
    actor,ref,engine,tokenizer=build_models(cfg,42)
    result=evaluate(engine,tokenizer,cfg,tasks,out/'metrics.json',200042,a.protocol)
    result['total_seconds_including_model_load']=time.monotonic()-start
    core.atomic_json(out/'summary.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
