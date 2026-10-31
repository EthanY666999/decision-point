"""Wait without CUDA, then acquire original supervisor lease and run read-only probes."""
import datetime,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from patr_core import atomic_json
OUT=ROOT/'runs/audit/state_probe_hist1';OUT.mkdir(parents=True,exist_ok=True)
PYTHON='/home/vipuser/patr-reproduction/.venv/bin/python'
def event(**kw):
    kw['unix']=time.time()
    with (OUT/'scheduler_events.jsonl').open('a') as f:f.write(json.dumps(kw)+'\n')
def gpu_pids():
    s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    return {int(x.strip()) for x in s.splitlines() if x.strip()}
def descendants(pid):
    pairs=[line.split() for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines()];found={pid}
    for _ in range(8):
        more={int(p) for p,pp in pairs if int(pp) in found}
        if more<=found:break
        found|=more
    return found
def main():
    lock=(OUT/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=json.loads((ROOT/'artifacts/deadline.json').read_text())['unix']
    lease=(ROOT/'runs/control_queue/supervisor.lock').open('a')
    atomic_json(OUT/'scheduler_status.json',dict(status='waiting_for_full_queue_completion',pid=os.getpid(),gpu_initialized=False,deadline_unix=deadline,gpu_overlap_observed=False))
    event(kind='wait_started',pid=os.getpid(),gpu_initialized=False)
    while time.time()<deadline-120:
        state=json.loads((ROOT/'runs/control_queue/status.json').read_text())
        if state.get('status')=='controls_complete':
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:time.sleep(10);continue
            if gpu_pids():fcntl.flock(lease,fcntl.LOCK_UN);time.sleep(10);continue
            event(kind='original_queue_complete_lease_acquired',queue_status=state,gpu_pids=[]);break
        time.sleep(20)
    else:
        atomic_json(OUT/'scheduler_status.json',dict(status='not_started_rental_deadline',gpu_initialized=False,gpu_overlap_observed=False));return
    old=Path('/home/vipuser/golden-sft-patr-20261031/code');ext=Path('/home/vipuser/golden-patr-extended800-20261031/code')
    models=[('old_patr',Path('/home/vipuser/patr-v1-tree-20261016/runs/formal/patr_seed42/final_model')),('sft',old/'runs/golden_sft/sft/final_model'),('sft_grpo400',old/'runs/formal/grpo_seed42/final_model'),('sft_patr400',old/'runs/formal/patr_seed42/final_model'),('sft_patr500',ext/'runs/formal/patr_seed42/final_model'),('sft_tree_random400',ROOT/'runs/formal/tree_random_seed42/final_model'),('base',Path('/home/vipuser/patr-reproduction/artifacts/Qwen2.5-0.5B-Instruct'))]
    for label,model in models:
        done=(OUT/label/'summary.json').exists() and (label!='sft_tree_random400' or (OUT/label/'no_history_reference/summary.json').exists())
        if done:continue
        if time.time()>=deadline-120:
            atomic_json(OUT/'scheduler_status.json',dict(status='paused_deadline',next_label=label,gpu_overlap_observed=False));return
        assert not gpu_pids(),'External GPU owner detected; no inference started'
        assert (model/'model.safetensors').exists()
        cmd=[PYTHON,'-u','scripts/state_probe_hist.py','--model',str(model),'--label',label]
        if label=='sft_tree_random400':cmd+=['--fresh-reference']
        env=dict(os.environ,HIST1_GPU_LEASE='queue_complete_lock_held',HIST1_GPU_LEASE_FD=str(lease.fileno()))
        with (OUT/(label+'.log')).open('a') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(lease.fileno(),))
            start=time.time();overlap=False;stopped=False
            atomic_json(OUT/'scheduler_status.json',dict(status='running',label=label,pid=child.pid,supervisor_pid=os.getpid(),gpu_overlap_observed=False,deadline_unix=deadline))
            event(kind='model_start',label=label,pid=child.pid,gpu_pids_before=[])
            while child.poll() is None:
                owners=gpu_pids();unexpected=owners-descendants(child.pid)
                if unexpected:
                    overlap=True;event(kind='unexpected_gpu_owner',pids=sorted(unexpected),label=label)
                if (time.time()>=deadline or unexpected) and not stopped:
                    child.send_signal(signal.SIGTERM);stopped=True
                event(kind='gpu_ownership_check',label=label,expected_pid=child.pid,gpu_pids=sorted(owners),unexpected=sorted(unexpected))
                time.sleep(10)
            event(kind='model_exit',label=label,returncode=child.returncode,seconds=time.time()-start,gpu_overlap_observed=overlap)
            if child.returncode or stopped:
                atomic_json(OUT/'scheduler_status.json',dict(status='paused_deadline' if time.time()>=deadline else 'needs_recovery',label=label,exit_code=child.returncode,gpu_overlap_observed=overlap));return
        subprocess.run([PYTHON,'scripts/state_probe_hist.py','--summarize'],cwd=ROOT,check=True)
        time.sleep(3)
    atomic_json(OUT/'scheduler_status.json',dict(status='complete',completed_models=len(models),gpu_overlap_observed=False,finished_unix=time.time()))
    event(kind='all_complete',gpu_overlap_observed=False)
if __name__=='__main__':main()
