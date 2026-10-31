"""Serialized GPU queue with exclusive lock, no duplicate launches, fixed rental deadline."""
import fcntl,json,math,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from patr_core import atomic_json
os.chdir(ROOT);OUT=ROOT/'runs/control_queue';OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PYTHON='/home/vipuser/patr-reproduction/.venv/bin/python';deadline=json.loads((ROOT/'artifacts/deadline.json').read_text())['unix']
old='/home/vipuser/golden-sft-patr-20261031/code';ext='/home/vipuser/golden-patr-extended800-20261031/code'
models=[('base','/home/vipuser/patr-reproduction/artifacts/Qwen2.5-0.5B-Instruct'),('old_patr','/home/vipuser/patr-v1-tree-20261016/runs/formal/patr_seed42/final_model'),('sft',old+'/runs/golden_sft/sft/final_model'),('sft_grpo400',old+'/runs/formal/grpo_seed42/final_model'),('sft_patr400',old+'/runs/formal/patr_seed42/final_model'),('sft_patr500',ext+'/runs/formal/patr_seed42/final_model')]
stages=[]
for name,model in models:
    stages.append((f'probe_{name}',[PYTHON,'-u','scripts/state_probe.py','--label',name,'--model',model],ROOT/f'runs/audit/state_probe/{name}/summary.json'))
for method,conf in [('tree_random','tree_random400'),('grpo','grpo4_400')]:
    stages.extend([(f'{method}_smoke',[PYTHON,'-u','train.py','--method',method,'--config',f'configs/{conf}.json','--smoke'],ROOT/f'runs/smoke/{method}_seed42/final_test.json'),(f'{method}_400',[PYTHON,'-u','train.py','--method',method,'--config',f'configs/{conf}.json'],ROOT/f'runs/formal/{method}_seed42/final_test.json'),(f'{method}_supplemental',[PYTHON,'-u','scripts/golden_supplemental.py','--model',str(ROOT/f'runs/formal/{method}_seed42/final_model'),'--output',str(ROOT/f'runs/audit/supplemental500/{method}')],ROOT/f'runs/audit/supplemental500/{method}/summary.json')])
def healthy_smoke(method):
    rows=[json.loads(x) for x in (ROOT/f'runs/smoke/{method}_seed42/train_metrics.jsonl').read_text().splitlines()]
    assert rows and all(math.isfinite(r[k]) for r in rows for k in ['loss','grad_norm','max_parameter_change'])
    assert any(r['max_parameter_change']>0 for r in rows),'No smoke parameter update; inspect groups before continuing'
for name,cmd,done in stages:
    if done.exists():continue
    if time.time()>=deadline-120:
        atomic_json(OUT/'status.json',dict(status='paused_deadline',next_stage=name));sys.exit(0)
    # Do not collide with any other CUDA process. This queue is the sole GPU owner.
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not gpu, f'GPU already occupied: {gpu}'
    if name.endswith('_400'):
        method=name[:-4];healthy_smoke(method)
        cp=ROOT/f'runs/formal/{method}_seed42/checkpoint.pt'
        if cp.exists():cmd.append('--resume')
    log=OUT/f'{name}.log'
    with log.open('a') as f:
        child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        atomic_json(OUT/'status.json',dict(status='running',stage=name,pid=child.pid,supervisor_pid=os.getpid(),command=cmd,started_unix=time.time(),deadline_unix=deadline))
        sent=False
        while child.poll() is None:
            if time.time()>=deadline and not sent:
                os.kill(child.pid,signal.SIGTERM);sent=True
                atomic_json(OUT/'status.json',dict(status='deadline_stop_requested',stage=name,pid=child.pid))
            time.sleep(10)
        if child.returncode!=0 or not done.exists():
            atomic_json(OUT/'status.json',dict(status='needs_recovery' if not sent else 'paused_deadline',stage=name,exit_code=child.returncode,log=str(log)));sys.exit(1)
    time.sleep(3)
atomic_json(OUT/'status.json',dict(status='controls_complete',finished_unix=time.time(),next='Analyze controls before backtracking; no automatic new training'))
