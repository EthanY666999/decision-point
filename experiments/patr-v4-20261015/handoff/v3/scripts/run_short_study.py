"""Finite preregistered short validation only. No formal training or automatic retry."""
import json,subprocess,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from patr_core import atomic_json
from run_integrity import sha256_file

out=ROOT/'runs/short_study'
out.mkdir(parents=True,exist_ok=True)
manifest=out/'study_status.json'
if manifest.exists():raise RuntimeError('Study already exists: inspect state; no silent rerun')
probe=ROOT/'reports/method_probe.json'
assert probe.exists() and json.loads(probe.read_text())['passed'],'All-method probe must pass first'
plan=[]
for method in ['grpo','patr']:
    for profile in ['legacy','behavior_variant_v2']:
        config='configs/short_'+profile+'.json'
        cfg=json.loads((ROOT/config).read_text())
        assert cfg['run_mode']=='short_validation' and cfg['steps']==20
        assert cfg['validation_tasks']==100 and cfg['dataset_path']=='artifacts/data_v2.json'
        run_root='runs/short_study/'+profile
        cmd=[sys.executable,'-u','train.py','--method',method,'--config',config,'--run-root',run_root]
        plan.append(dict(method=method,profile=profile,config=config,config_sha256=sha256_file(ROOT/config),command=cmd))
state=dict(status='running',started_unix=time.time(),pid=os.getpid(),plan=plan,completed=[],scope='4 fixed 20-step runs; 100 validation tasks x4; never test/holdout/formal')
atomic_json(manifest,state)
for item in plan:
    state['current']=item;atomic_json(manifest,state)
    log=ROOT/'logs'/('short_'+item['method']+'_'+item['profile']+'.log')
    begin=time.time()
    with log.open('w') as f:
        result=subprocess.run(item['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    row=dict(**item,returncode=result.returncode,wall_seconds=time.time()-begin,log=str(log))
    state['completed'].append(row)
    status=ROOT/'runs/short_study'/item['profile']/(item['method']+'_seed42')/'status.json'
    if result.returncode!=0 or not status.exists() or json.loads(status.read_text())['status']!='complete':
        state.update(status='failed_or_paused',finished_unix=time.time());atomic_json(manifest,state)
        raise RuntimeError('Short run did not complete; no automatic retry or formal continuation')
    atomic_json(manifest,state)
state.update(status='complete',current=None,finished_unix=time.time(),wall_seconds=time.time()-state['started_unix'])
atomic_json(manifest,state)
print(json.dumps(state),flush=True)
