"""Start the formal queue only after every preflight and full-batch calibration pass."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patr_core import ROOT,atomic_json
limit=time.time()+7200
atomic_json(ROOT/'artifacts/queue_status.json',dict(status='waiting_for_preflight',unix=time.time()))
while subprocess.run(['tmux','has-session','-t','patr-async-validate'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
    if time.time()>limit:raise TimeoutError('Preflight exceeded two hours; formal queue not started')
    time.sleep(15)
for method in ['grpo','patr','tree_random','dapo','arpo']:
    p=ROOT/'runs/smoke'/f'{method}_seed42'/'status.json'
    assert p.exists() and json.loads(p.read_text())['status']=='smoke_complete', f'{method} preflight incomplete'
p=ROOT/'runs/formal/grpo_seed42/status.json'
s=json.loads(p.read_text())
assert s['status']=='paused_checkpointed' and s['step']==1,'Full-batch calibration or checkpoint failed'
subprocess.run([sys.executable,'scripts/cleanup_smoke_weights.py','--apply'],cwd=ROOT,check=True)
print('PREFLIGHT_PASSED_STARTING_FORMAL_QUEUE',flush=True)
raise SystemExit(subprocess.call([sys.executable,'-u','scripts/run_queue.py'],cwd=ROOT))
