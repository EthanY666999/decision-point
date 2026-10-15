"""Launch a new formal run, never resume or restart implicitly."""
import json,os,subprocess,time
from pathlib import Path
root=Path('/home/vipuser/patr-reproduction-v4')
snapshot=root/'snapshots/smoke-20261015-a'
smoke=root/'runs/smoke-20261015-a/summary.json'
x=json.loads(smoke.read_text());assert x['completed'] and x['restore_verified'] and x['counts']['optimizer_updates']==2
assert x['implementation_fingerprint_sha256']==json.loads((snapshot/'frozen_identity.json').read_text())['sha256']
assert time.time()<json.loads((snapshot/'config_candidate.json').read_text())['deadline_unix']-900
assert os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>25*1024**3
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
out=root/'runs/patr-400-20261015-a';assert not out.exists()
log=(root/'reports/patr_400_execution.log').open('xb')
args=['/home/vipuser/patr-reproduction/.venv/bin/python','-u',str(snapshot/'train_v4.py'),'--phase','patr','--out',str(out),'--r1-summary',str(root/'runs/r1-20261015-a/summary.json'),'--smoke-summary',str(smoke)]
p=subprocess.Popen(args,cwd=snapshot,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,phase='patr',target=400,engineering_gate=40,snapshot=str(snapshot),out=str(out),args=args,started_unix=time.time(),state='launched_awaiting_initial_evaluation',independent_base_model=True)
(root/'reports/patr_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
