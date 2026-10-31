"""Back up completed new smoke artifacts before any separately verified cleanup."""
import json
import subprocess
import time
from pathlib import Path
import backup_completed_local as backup

root = '/home/vipuser/golden-next-20261031/seed43_patr/code/runs/smoke/patr_seed43'
backup.OUT = Path('/Users/ethan/Documents/ChatGPT/ACL2027/output/golden-next-20261031/weights-backup-seed43-patr-smoke')
backup.ROOTS = {'seed43_patr_smoke': root}
if __name__ == '__main__':
    state = json.loads(subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'galaxy-4090', 'cat '+root+'/status.json'], text=True))
    assert state['status'] == 'smoke_complete' and state['step'] == 1, state
    try:
        backup.main()
    except Exception as exc:
        if backup.OUT.exists():
            backup.save(backup.OUT/'status.json', dict(status='needs_recovery', error=repr(exc), time=time.time()))
        raise
