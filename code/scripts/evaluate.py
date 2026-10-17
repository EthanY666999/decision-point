"""Evaluate a local final_model using the original 100-task protocol (no training)."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--model',required=True,type=Path)
p.add_argument('--output',required=True,type=Path);p.add_argument('--split',choices=['validation','test'],default='test')
a=p.parse_args();model=a.model.resolve();output=a.output.resolve()
if output.exists() or output.with_suffix('.jsonl').exists():raise SystemExit('Output exists; choose a new filename.')
if not (model/'model.safetensors').exists():raise SystemExit('Supply a local exported final_model directory (weights are not included in Git).')
# This import sets the same environment defaults as the archived trainer.
from train import build_models,evaluate
from patr_core import make_datasets
cfg=json.loads((ROOT/'configs/frozenlake_05b.json').read_text())
cfg.update(json.loads((ROOT/'artifacts/runtime_override.json').read_text()));cfg['model']=str(model)
actor,ref,engine,tokenizer=build_models(cfg,cfg['seed'])
print(json.dumps(evaluate(engine,tokenizer,cfg,make_datasets()[a.split],output,
                         200042 if a.split=='test' else 100042,a.split),indent=2))
