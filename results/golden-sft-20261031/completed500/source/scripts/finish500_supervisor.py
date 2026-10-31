"""Honor the user's reduced budget after the previous trainer safely checkpoints."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patr_core import atomic_json

out = ROOT / 'runs/finish500'
out.mkdir(parents=True, exist_ok=True)
lock = (out / 'supervisor.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

def status(stage, state='running', **kw):
    atomic_json(out / 'status.json', dict(stage=stage, status=state, pid=os.getpid(),
                                        updated_unix=time.time(), **kw))

try:
    status('waiting_for_safe_checkpoint')
    request = json.loads((ROOT / 'runs/extension/user_stop_at_500.json').read_text())
    old_pid = request['observed_status']['pid']
    while Path(f'/proc/{old_pid}').exists():
        if Path(f'/proc/{old_pid}/stat').read_text().split()[2] == 'Z':
            break
        time.sleep(3)
    saved = json.loads((ROOT / 'runs/formal/patr_seed42/status.json').read_text())
    assert saved['status'] == 'paused_checkpointed' and saved['step'] <= 500, saved
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
    status('training_to_500', resumed_step=saved['step'])
    with (out / 'train.log').open('x') as log:
        subprocess.run([sys.executable, '-u', 'scripts/run.py', '--method', 'patr',
                        '--config', 'configs/golden_finish500.json', '--resume', '--reduce-training-budget'],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    done = json.loads((ROOT / 'runs/formal/patr_seed42/status.json').read_text())
    assert done['status'] == 'complete' and done['step'] == 500, done
    status('diagnostics')
    with (out / 'diagnostics.log').open('x') as log:
        subprocess.run([sys.executable, 'scripts/golden_diagnostics.py', '--run',
                        str(ROOT / 'runs/formal/patr_seed42')], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    status('complete_at_user_limit', 'complete', step=500, additional_training_authorized=False)
except Exception as exc:
    status('failure', 'failed', error=repr(exc))
    raise
