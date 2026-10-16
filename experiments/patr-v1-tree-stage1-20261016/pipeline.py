import json,time,subprocess,os,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent;py='/home/vipuser/patr-reproduction/.venv/bin/python'
def status(**v):
 p=r/'reports/pipeline_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(time=time.time(),**v),indent=2));t.replace(p)
# Existing deadline authorization remains binding: Oct17 04:45 Beijing save guard.
deadline=1792183500
status(phase='sampling_gate_wait',training_started=False)
while not (r/'reports/stage1_summary.json').exists():
 if time.time()>deadline:status(phase='deadline_before_training');raise SystemExit
 time.sleep(30)
gate=json.loads((r/'reports/stage1_summary.json').read_text())
if not gate['passed']:status(phase='gate_failed_no_training',gate=gate['comparison']);raise SystemExit
assert (r/'train.py').read_bytes()==Path('/home/vipuser/patr-reproduction/train.py').read_bytes()
identity={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ('train.py','patr_core.py','configs/frozenlake_05b.json','artifacts/data.json','artifacts/runtime_override.json')}
(r/'reports/frozen_source.json').write_text(json.dumps(identity,indent=2))
(r/'artifacts/deadline.json').write_text(json.dumps({'unix':deadline,'reason':'Existing user deadline Oct17 05:00 Beijing; save fifteen minutes before'}))
status(phase='training_patr400',training_started=True)
with (r/'reports/patr400.log').open('w') as f:
 p=subprocess.Popen([py,'-u','train.py','--method','patr'],cwd=r,stdout=f,stderr=subprocess.STDOUT)
 status(phase='training_patr400',pid=p.pid,training_started=True)
 rc=p.wait()
status(phase='training_exited',returncode=rc,training_started=True,run_status=json.loads((r/'runs/formal/patr_seed42/status.json').read_text()) if (r/'runs/formal/patr_seed42/status.json').exists() else None)
