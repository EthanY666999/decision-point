from pathlib import Path
import json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1];old=Path('/home/vipuser/golden-sft-patr-20261031/code')
cfg=json.loads((old/'runs/formal/patr_seed42/config.json').read_text());cfg.pop('method')
# No algorithm or reward edits; run label also retained for strict comparison.
(ROOT/'configs/tree_random400.json').write_text(json.dumps(cfg,indent=2))
g4=dict(cfg,flat_group_size=4);(ROOT/'configs/grpo4_400.json').write_text(json.dumps(g4,indent=2))
deadline=datetime.datetime(2026,11,1,4,55,tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
(ROOT/'artifacts/deadline.json').write_text(json.dumps({'unix':deadline.timestamp(),'cst':deadline.isoformat()}))
# mmap affects host memory only and permits safe resume on the16GiB host.
p=ROOT/'train.py';s=p.read_text();s=s.replace("torch.load(cp,map_location='cpu',weights_only=False)","torch.load(cp,map_location='cpu',weights_only=False,mmap=True)");p.write_text(s)
plan={'authorized':'User now says follow Claude recommendations; supersedes old prohibition on NEW experiments. Original PaTR500 remains finished; never resume it.', 'order':['CPU branching and geometry audit','shared-state diagnostic of6 models','TreeRandom smoke then400 with same originalSFT checkpoint','fixed main100+supplemental500 evaluation','GRPO group4 smoke then400 if rental time permits; otherwise save/pause','review evidence and prior work before backtracking pilot'], 'tree_random_caveat':'Existing method disables score pruning as well as random expansion. Quantify actual score-pruning opportunity; do not call pure selection ablation unless zero.', 'grpo4_caveat':'Group-size control, not exactly compute-matched; report actual generated tokens, interactions and time. Do not claim equal compute.', 'backtracking_gate':'No automatic new-method launch before audited sampling/masking semantics, predeclared100step3arm protocol and validation/signal gate. BranPO already backtracks incorrect trajectories; innovation cannot be backtracking alone.', 'test_policy':'Test probes and filtered409 are diagnostic only, no hyperparameter selection. Training selection uses originalvalidation; test reports original100 and500 separately with409 sensitivity.', 'deadline':deadline.isoformat(),'hardware':'RTX4090 48GiB; old400 onA100; wall-clock comparison not fair', 'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['train.py','patr_core.py']},'starting_model':cfg['model']}
(ROOT/'plan.json').write_text(json.dumps(plan,indent=2))
