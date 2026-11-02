"""CPU waiter for the audited B2 pair, strictly after C minimum completion."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NEXT = ROOT.parents[1] / 'code'
CONTROLS = Path('/home/vipuser/golden-controls-20261031/code')
OUT = ROOT / 'runs/queue'
PY = '/home/vipuser/patr-reproduction/.venv/bin/python'
METHODS = ('patr', 'tree_random')


def read(p):
    return json.loads(p.read_text())


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(p)


def deadline():
    return read(CONTROLS / 'artifacts/deadline.json')['unix']


def owners():
    return {int(x) for x in subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()}


def descendants(pid):
    result = {pid}
    for _ in range(8):
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():
                continue
            try:
                if int((p / 'stat').read_text().rsplit(')', 1)[1].split()[1]) in result:
                    result.add(int(p.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    return result


def prerequisites():
    return (read(NEXT / 'runs/queue/status.json').get('status') == 'A_B1_Cminimum_complete'
            and read(CONTROLS / 'runs/control_queue/status.json').get('status') == 'controls_complete'
            and read(CONTROLS / 'runs/audit/state_probe_hist1/scheduler_status.json').get('status') == 'complete')


def complete(method, smoke):
    p = ROOT / 'runs' / ('smoke' if smoke else 'formal') / f'{method}_seed42'
    if not (p / 'status.json').exists():
        return False
    state = read(p / 'status.json')
    return (state.get('status') == ('smoke_complete' if smoke else 'complete')
            and state.get('step') == state.get('total_steps') == (1 if smoke else 150)
            and (p / 'final_test.json').exists())


def audit_smoke(method):
    p = ROOT / 'runs/smoke' / f'{method}_seed42'
    assert complete(method, True)
    identity = read(p / 'b2_identity.json')
    assert identity['smoke'] and identity['method'] == method
    assert identity['config'] == read(ROOT / 'configs/b2_k2_150.json')
    for name, sha in identity['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    rows = [json.loads(line) for line in (p / 'train_metrics.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['step'] == 1
    row = rows[0]
    assert all(math.isfinite(row[k]) for k in ('loss', 'grad_norm', 'max_parameter_change'))
    assert row['max_parameter_change'] > 0 and 0 < row['max_leaves'] <= 8
    assert row['leaf_cap'] == 8 and row['task_groups'] == 4
    save(p / 'technical_acceptance.json', dict(passed=True, time=time.time(),
         finite_gradients=True, parameter_update=True, leaf_cap=8,
         independent_sft=identity['config']['model'], no_reward_threshold=True))


def stages():
    return [(m, smoke) for m in METHODS for smoke in (True, False)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    own = (OUT / 'scheduler.lock').open('a')
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(ROOT / 'artifacts/b2_protocol.json')
    gate_path = NEXT / 'runs/b1/gate.json'
    assert hashlib.sha256(gate_path.read_bytes()).hexdigest() == protocol['B1_gate_sha256']
    assert protocol['B1_pass'] and protocol['selected_K'] == 2
    assert hashlib.sha256((ROOT / 'artifacts/data.json').read_bytes()).hexdigest() == protocol['data_sha256']
    lease = (CONTROLS / 'runs/control_queue/supervisor.lock').open('a')
    save(OUT / 'status.json', dict(status='waiting_for_Cminimum', pid=os.getpid(), gpu_initialized=False))
    while time.time() < deadline() - 300:
        if prerequisites():
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(20)
                continue
            if not owners():
                break
            fcntl.flock(lease, fcntl.LOCK_UN)
        time.sleep(20)
    else:
        save(OUT / 'status.json', dict(status='deadline_before_admission'))
        return
    for method, smoke in stages():
        if complete(method, smoke):
            if smoke:
                audit_smoke(method)
            continue
        remaining = sum(900 if s else 12600 for m, s in stages() if not complete(m, s)) + 900
        free = shutil.disk_usage(ROOT).free
        if deadline() - time.time() < remaining or free < 16 * 1024**3:
            save(OUT / 'status.json', dict(status='capacity_pending', method=method, smoke=smoke,
                 required_seconds=remaining, available_seconds=deadline()-time.time(), free_bytes=free,
                 recovery='Verify completed new-run backups before cleanup; never delete active or old archives.'))
            return
        if not smoke:
            audit_smoke(method)
        run = ROOT / 'runs' / ('smoke' if smoke else 'formal') / f'{method}_seed42'
        command = [PY, '-u', 'run_b2.py', '--method', method, '--config', 'configs/b2_k2_150.json']
        if smoke:
            command.append('--smoke')
        if (run / 'checkpoint.pt').exists():
            command.append('--resume')
        assert not owners(), 'Unexpected GPU owner; refuse duplicate launch'
        name = f'{method}_{"smoke" if smoke else "150"}'
        with (OUT / f'{name}.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lease.fileno(),),
                env=dict(os.environ, NEXT_GPU_LEASE_FD=str(lease.fileno())))
            save(OUT / 'status.json', dict(status='running', stage=name, pid=child.pid,
                 supervisor_pid=os.getpid(), started_unix=time.time(), command=command))
            sent = False
            while child.poll() is None:
                unexpected = owners() - descendants(child.pid)
                if unexpected or time.time() >= deadline():
                    if not sent:
                        child.send_signal(signal.SIGTERM)
                        sent = True
                    save(OUT / 'intervention.json', dict(time=time.time(), stage=name,
                         unexpected_gpu_pids=sorted(unexpected), deadline_reached=time.time() >= deadline()))
                time.sleep(10)
        if child.returncode or not complete(method, smoke):
            save(OUT / 'status.json', dict(status='needs_recovery', stage=name, returncode=child.returncode))
            return
        if smoke:
            audit_smoke(method)
        with (OUT / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=name, completed_unix=time.time())) + '\n')
        time.sleep(3)
    assert not owners()
    subprocess.run([PY, 'scripts/b2_gate.py', '--patr-run', 'runs/formal/patr_seed42',
        '--random-run', 'runs/formal/tree_random_seed42', '--output', 'runs/analysis/gate.json'], cwd=ROOT, check=True)
    save(OUT / 'status.json', dict(status='B2_complete', finished_unix=time.time(),
         gate=read(ROOT / 'runs/analysis/gate.json'), next='Report gate; no automatic B3 or backtracking launch.'))


if __name__ == '__main__':
    main()
