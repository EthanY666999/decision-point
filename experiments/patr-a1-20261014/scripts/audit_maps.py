import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patr_core import ROOT,make_datasets,make_env,distance,atomic_json
from collections import Counter
cfg=json.loads((ROOT/'configs/frozenlake_05b.json').read_text())
data=make_datasets(); hashes={}; result={}
for split,tasks in data.items():
    fingerprints=[];ds=[]
    for task in tasks:
        e=make_env(task,cfg)
        fingerprints.append(hashlib.sha256(e.desc.tobytes()).hexdigest())
        ds.append(distance(e))
    hashes[split]=set(fingerprints)
    result[split]=dict(n=len(tasks),unique_maps=len(hashes[split]),shortest_path_counts=dict(Counter(ds)))
result['shared_geometry_train_test']=len(hashes['train']&hashes['test'])
result['shared_geometry_train_validation']=len(hashes['train']&hashes['validation'])
result['note']='Upstream task generation retained unchanged. Distinct UIDs can share small-map geometry; this is not a geometry-held-out generalization test.'
atomic_json(ROOT/'artifacts/map_audit.json',result)
print(json.dumps(result))
