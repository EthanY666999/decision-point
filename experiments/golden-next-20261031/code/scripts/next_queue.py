"""CPU waiter -> A -> B1 -> priority minimum C; one inherited GPU lease.

Never interrupts the existing controls/history queue. Later gated B2/B3 and D
are deliberately not auto-launched by this queue before evidence is reviewed.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
C=Path('/home/vipuser/golden-controls-20261031/code')
PY='/home/vipuser/patr-reproduction/.venv/bin/python'
OUT=ROOT/'runs/queue'

def atomic(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)

def deadline():return json.loads((C/'artifacts/deadline.json').read_text())['unix']
def owners():return {int(p) for p in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()}
def descendants(pid):
    result={pid}
    for _ in range(5):
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                st=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(st[1]) in result:result.add(int(p.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return result

def stages():
    s=[]
    for label in ['sft_grpo400','sft_patr400','sft_patr500','sft_tree_random400']:
        s.append((f'A1_{label}',ROOT,[PY,'-u','scripts/evaluate_additional.py','--label',label,'--protocol','map8'],ROOT/f'runs/evaluations/map8/{label}/summary.json',900,False))
    s.append(('A2_cross_gpu',ROOT,[PY,'-u','scripts/evaluate_additional.py','--label','sft_patr400','--protocol','cross_gpu_main100'],ROOT/'runs/evaluations/cross_gpu_main100/sft_patr400/summary.json',900,False))
    s.append(('A3456_summary',ROOT,[PY,'scripts/summarize_additional.py'],ROOT/'runs/analysis/A_complete.json',120,False))
    for model in ['sft','sft_grpo400']:
        s.append((f'B1_{model}',ROOT,[PY,'-u','scripts/branch_frequency.py','--model',model],ROOT/f'runs/b1/{model}/summary.json',5400,False))
    s.append(('B1_gate',ROOT,[PY,'scripts/summarize_b1.py'],ROOT/'runs/b1/gate.json',120,False))
    for method in ['patr','tree_random']:
        d=ROOT.parent/f'seed43_{method}/code'
        s.append((f'C_{method}_43_smoke',d,[PY,'-u','train.py','--method',method,'--config','configs/seed43.json','--smoke'],d/f'runs/smoke/{method}_seed43/final_test.json',900,False))
        s.append((f'C_{method}_43_400',d,[PY,'-u','train.py','--method',method,'--config','configs/seed43.json'],d/f'runs/formal/{method}_seed43/final_test.json',21600,True))
        s.append((f'C_{method}_43_supplemental',d,[PY,'-u','scripts/golden_supplemental.py','--model',str(d/f'runs/formal/{method}_seed43/final_model'),'--output',str(d/'runs/supplemental500')],d/'runs/supplemental500/summary.json',900,False))
    return s

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    own=(OUT/'scheduler.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lease=(C/'runs/control_queue/supervisor.lock').open('a')
    atomic(OUT/'status.json',dict(status='waiting_for_controls_and_history',pid=os.getpid(),gpu_initialized=False,deadline_unix=deadline()))
    while time.time()<deadline()-120:
        cs=json.loads((C/'runs/control_queue/status.json').read_text())
        hs=json.loads((C/'runs/audit/state_probe_hist1/scheduler_status.json').read_text())
        if cs.get('status')=='controls_complete' and hs.get('status')=='complete':
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:time.sleep(10);continue
            if not owners():break
            fcntl.flock(lease,fcntl.LOCK_UN)
        time.sleep(20)
    else:
        atomic(OUT/'status.json',dict(status='paused_deadline_before_start'));return
    for name,cwd,cmd,done,estimate,training in stages():
        if done.exists():continue
        remaining=deadline()-time.time()
        if remaining<estimate+300:
            atomic(OUT/'status.json',dict(status='insufficient_window',next_stage=name,remaining_seconds=remaining,required_seconds=estimate+300));return
        if training:
            if shutil.disk_usage(cwd).free<16*1024**3:
                atomic(OUT/'status.json',dict(status='needs_verified_backup_for_disk',next_stage=name,free_bytes=shutil.disk_usage(cwd).free));return
            cp=done.parent/'checkpoint.pt'
            if cp.exists():cmd=cmd+['--resume']
            # Smoke must be technically sound; do not demand positive reward in one batch.
            method='tree_random' if 'tree_random' in name else 'patr'
            smoke=cwd/f'runs/smoke/{method}_seed43/train_metrics.jsonl'
            import math
            rows=[json.loads(l) for l in smoke.read_text().splitlines()]
            assert rows and all(math.isfinite(r[k]) for r in rows for k in ['loss','grad_norm','max_parameter_change'])
        assert not owners(), 'Unexpected GPU process; no duplicate launch'
        # Incomplete inference evidence must be moved aside explicitly by recovery, never overwritten.
        env=dict(os.environ,NEXT_GPU_LEASE_FD=str(lease.fileno()))
        with (OUT/f'{name}.log').open('a') as log:
            p=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True,pass_fds=(lease.fileno(),))
            start=time.time();sent=False
            atomic(OUT/'status.json',dict(status='running',stage=name,pid=p.pid,supervisor_pid=os.getpid(),command=cmd,cwd=str(cwd),started_unix=start,deadline_unix=deadline()))
            while p.poll() is None:
                unexpected=owners()-descendants(p.pid)
                if unexpected or time.time()>=deadline():
                    if not sent:p.send_signal(signal.SIGTERM);sent=True
                    atomic(OUT/'intervention.json',dict(stage=name,unexpected_gpu_pids=sorted(unexpected),deadline_reached=time.time()>=deadline(),time=time.time()))
                time.sleep(10)
        if name=='A3456_summary' and p.returncode==0:
            atomic(done,dict(status='complete',report='additional_comparison.json',finished_unix=time.time()))
        if p.returncode or not done.exists():
            atomic(OUT/'status.json',dict(status='needs_recovery',stage=name,returncode=p.returncode,done_exists=done.exists(),log=str(OUT/f'{name}.log')));return
        with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(dict(stage=name,status='complete',seconds=time.time()-start,finished_unix=time.time()))+'\n')
        time.sleep(3)
    atomic(OUT/'status.json',dict(status='A_B1_Cminimum_complete',finished_unix=time.time(),next='Review B1 training-only gate and remaining capacity; implement audited B2 if passed, else continue C. No automatic backtracking experiment.'))

if __name__=='__main__':main()
