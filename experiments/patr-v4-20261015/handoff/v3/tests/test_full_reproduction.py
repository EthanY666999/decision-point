"""The explicitly authorized full mode evaluates reserved test only at completion."""
import json,random,sys
from types import SimpleNamespace
import numpy as np
import torch
import train as tr
from test_final_resume import short_fixture

def test_full_resume_at_400_evaluates_test_and_exports_model(tmp_path,monkeypatch):
 cfg,data=short_fixture(tmp_path,monkeypatch)
 cfg.update(steps=400,run_mode='formal',formal_authorization='user_full_rerun_20261015')
 (tmp_path/'configs/short_test.json').write_text(json.dumps(cfg))
 run=tmp_path/'runs/short_validation/grpo_seed42';run.mkdir(parents=True);(run/'checkpoint.pt').touch()
 monkeypatch.setattr(tr,'prepare_run',lambda *a,**k:dict(start_step=400,recovery=None))
 state=dict(config=cfg,model={},optimizer={},step=400,rng=np.random.default_rng(42).bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=[],python_rng=random.getstate())
 saved=[];ev=[]
 actor=SimpleNamespace(parameters=lambda:[],load_state_dict=lambda _:None,save_pretrained=lambda p,**k:saved.append(('model',p.name)))
 tok=SimpleNamespace(save_pretrained=lambda p:saved.append(('tokenizer',p.name)))
 monkeypatch.setattr(tr,'STOP',False)
 monkeypatch.setattr(tr,'build_models',lambda *a:(actor,None,None,tok))
 monkeypatch.setattr(tr,'sync_weights',lambda *a:None)
 monkeypatch.setattr(tr.torch,'load',lambda *a,**k:state)
 monkeypatch.setattr(tr.torch.optim,'AdamW',lambda *a,**k:SimpleNamespace(load_state_dict=lambda _:None))
 monkeypatch.setattr(tr.torch.cuda,'set_rng_state_all',lambda _:None)
 def evaluate(e,t,c,tasks,path,seed,split):
  assert tasks==data['test'];assert path.name=='final_test.json';assert split=='test'
  ev.append(split);return {'n_tasks':100,'n_trajectories':400}
 monkeypatch.setattr(tr,'evaluate',evaluate)
 monkeypatch.setattr(sys,'argv',['train.py','--method','grpo','--config','configs/short_test.json','--resume'])
 tr.main()
 assert ev==['test'];assert saved==[('model','final_model'),('tokenizer','final_model')]
 assert json.loads((run/'status.json').read_text())['run_mode']=='formal'
