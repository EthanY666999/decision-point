import time,json,sys,os
from pathlib import Path
root=Path('/home/vipuser/patr-reproduction-v43-xformers');out=root/'runs/patr-400-xformers-a'
end=time.time()+300
while Path('/proc/984752').exists() and time.time()<end:time.sleep(2)
assert not Path('/proc/984752').exists(),'Trainer still active; inspect before any further action'
s=json.loads((out/'status.json').read_text());assert s['state']=='paused_saved',s
sys.path.insert(0,str(root));import torch;torch.set_num_threads(4)
from checkpoint_v4 import verified_load
state=verified_load(out/'latest.pt');assert state['fingerprint']==json.loads((out/'fingerprint.json').read_text())
assert state['counters']==s['counts']
def finite(x):
 if isinstance(x,torch.Tensor):return not x.is_floating_point() or bool(torch.isfinite(x).all())
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
 return True
assert finite(state['model']) and finite(state['optimizer'])
cp=out/'latest.pt';keep=out/'user-stopped-checkpoint.pt'
assert not keep.exists();os.link(cp,keep)
import shutil
shutil.copy2(str(cp)+'.manifest.json',str(keep)+'.manifest.json')
counts=dict(s['counts']);counts.pop('guard_state',None)
report=dict(state='stopped_by_user_saved_verified',time=time.time(),counts=counts,checkpoint=str(cp),preserved_checkpoint=str(keep),sha256=json.loads(Path(str(cp)+'.manifest.json').read_text())['sha256'],finite_model_optimizer=True,fingerprint_verified=True,training_pid_exited=True,controller_pid_exited=not Path('/proc/984618').exists(),final_test='not run: user stopped before 400',automatic_resume_allowed=False)
(root/'reports/user_stop_audit.json').write_text(json.dumps(report,indent=2))
(root/'reports/v43_pipeline_status.json').write_text(json.dumps(report,indent=2))
with (root/'reports/fault_handling.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),action='user_stop_completed',result=report))+'\n')
print(json.dumps(report))
