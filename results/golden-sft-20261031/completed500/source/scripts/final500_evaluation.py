"""Supplemental final evaluation only, after training has stopped at the user limit."""
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

OUT = ROOT / 'runs/finish500'
lock = (OUT / 'final_evaluation.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

def status(state, **kw):
    atomic_json(OUT / 'supplemental_status.json', dict(status=state, pid=os.getpid(),
                updated_unix=time.time(), training=False, **kw))

try:
    status('waiting_for_training_and_export')
    training_lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(training_lock, fcntl.LOCK_EX)
    done = json.loads((OUT / 'status.json').read_text())
    run = ROOT / 'runs/formal/patr_seed42'
    finished = json.loads((run / 'status.json').read_text())
    assert done['status'] == finished['status'] == 'complete'
    assert finished['step'] == finished['total_steps'] == 500
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
    script = Path('/home/vipuser/golden-sft-patr-20261031/code/scripts/golden_supplemental.py')
    status('evaluating', checkpoint_step=500)
    with (OUT / 'supplemental500.log').open('x') as log:
        subprocess.run([sys.executable, '-u', str(script), '--model', str(run/'final_model'),
                        '--output', str(OUT/'supplemental500')], stdout=log,
                       stderr=subprocess.STDOUT, cwd=script.parents[1], check=True)
    status('complete', checkpoint_step=500)
except Exception as exc:
    status('failed', error=repr(exc))
    raise
