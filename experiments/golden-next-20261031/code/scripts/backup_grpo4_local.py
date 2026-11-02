"""Verified immutable backup of the completed GRPO group-four run; Mac only."""
import json
import subprocess
import time
from pathlib import Path
import backup_completed_local as backup

backup.OUT = Path('/Users/ethan/Documents/ChatGPT/ACL2027/output/golden-next-20261031/weights-backup-grpo4')
root = '/home/vipuser/golden-controls-20261031/code/runs/formal/grpo_seed42'
backup.ROOTS = {'sft_grpo4_400': root}
if __name__ == '__main__':
    state = json.loads(subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'galaxy-4090', 'cat '+root+'/status.json'], text=True))
    assert state['status'] == 'complete' and state['step'] == 400, state
    try:
        backup.main()
    except Exception as exc:
        if backup.OUT.exists():
            backup.save(backup.OUT/'status.json', dict(status='needs_recovery', error=repr(exc), time=time.time()))
        raise
