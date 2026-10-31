"""CPU-only handover: preserve the existing trainer, then resume the old queue.

The caller stops only the old parent after verifying its identity. No process
group signals, no CUDA imports, and no second trainer while the adopted PID lives.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def identity(pid):
    try:
        root = Path('/proc') / str(pid)
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(start_ticks=stat[19], state=stat[0],
                    command=(root / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except FileNotFoundError:
        return None

def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--start-ticks', required=True)
    a = p.parse_args()
    out = a.root / 'runs/control_queue'
    with (out / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            live = identity(a.pid)
            if not live or live['state'] == 'Z':
                break
            assert live['start_ticks'] == a.start_ticks, 'PID reused; refuse adoption'
            assert 'train.py --method grpo --config configs/grpo4_400.json' in live['command']
            deadline = json.loads((a.root / 'artifacts/deadline.json').read_text())
            status = json.loads((out / 'status.json').read_text())
            status.update(supervisor_pid=os.getpid(), adopted_pid=a.pid,
                          deadline_unix=deadline['unix'], checked_unix=time.time())
            atomic(out / 'status.json', status)
            atomic(out / 'adoption_status.json', dict(status='adopted_existing_trainer',
                   trainer=live, pid=a.pid, supervisor_pid=os.getpid(),
                   checked_unix=time.time(), deadline=deadline))
            time.sleep(10)
        # A CUDA process can survive its host briefly during cleanup. Never race it.
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True).strip():
            time.sleep(10)
        atomic(out / 'adoption_status.json', dict(status='handover_to_original_queue',
               pid=a.pid, finished_unix=time.time()))
    os.chdir(a.root)
    os.execv(sys.executable, [sys.executable, '-u', 'scripts/controls_supervisor.py'])

if __name__ == '__main__':
    main()
