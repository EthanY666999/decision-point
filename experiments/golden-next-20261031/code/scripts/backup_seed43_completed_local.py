"""Mac-only completion waiter and SHA-verified backup of a new seed43 run.

Never copies an active checkpoint or deletes remote files. Shared existing
backup engine limits bandwidth and records per-file verified receipts.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
import backup_completed_local as backup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['patr', 'tree_random'], required=True)
    parser.add_argument('--stage', choices=['formal', 'smoke'], required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    label = f'seed43_{args.method}_{args.stage}'
    root = f'/home/vipuser/golden-next-20261031/seed43_{args.method}/code/runs/{args.stage}/{args.method}_seed43'
    backup.OUT = Path('/Users/ethan/Documents/ChatGPT/ACL2027/output/golden-next-20261031') / f'weights-backup-{label}'
    backup.ROOTS = {label: root}
    backup.OUT.mkdir(parents=True, exist_ok=True)
    lock = (backup.OUT / 'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = '''import json
from pathlib import Path
r=Path(ROOT)
s=json.loads((r/'status.json').read_text()) if (r/'status.json').exists() else {}
print(json.dumps(dict(state=s, checkpoint=(r/'checkpoint.pt').exists(), model=(r/'final_model/config.json').exists())))
'''.replace('Path(ROOT)', 'Path(' + repr(root) + ')')
    try:
        # Do not leave a completion waiter beyond the verified rental expiry.
        while time.time() < 1793603384:
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                'galaxy-4090', 'python3 -'], input=code, text=True, capture_output=True, check=True)
            info = json.loads(result.stdout)
            state = info['state']
            expected = 'complete' if args.stage == 'formal' else 'smoke_complete'
            count = 400 if args.stage == 'formal' else 1
            if state.get('status') == expected and state.get('step') == state.get('total_steps') == count:
                assert info['checkpoint'] and info['model'], 'Completed artifacts missing; do not create empty backup'
                backup.main()
                return
            assert args.wait, 'Run not complete; active checkpoints must not be copied'
            backup.save(backup.OUT / 'status.json', dict(status='waiting_for_completion', label=label,
                        observed_state=state, time=time.time()))
            time.sleep(20)
        raise RuntimeError('Rental expiry reached before completed backup admission')
    except Exception as exc:
        backup.save(backup.OUT / 'status.json', dict(status='needs_recovery', error=repr(exc), time=time.time()))
        raise


if __name__ == '__main__':
    main()
