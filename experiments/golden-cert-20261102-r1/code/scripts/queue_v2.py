"""CPU waiter: complete B3 first, exclusive lease, fresh platform evidence, gated stages."""
import fcntl,json,os,shutil,signal,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'runs/queue';Q.mkdir(parents=True,exist_ok=True)
CONTROL=Path('/home/vipuser/golden-controls-20261031/code')
B3=Path('/home/vipuser/golden-next-20261031/b3/code')
PY='/home/vipuser/patr-reproduction/.venv/bin/python'

def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def status(s,**kw):atomic(Q/'status.json',dict(status=s,supervisor_pid=os.getpid(),updated=time.time(),**kw))
def identity(pid):
    try:
        x=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if x[0]=='Z' else int(x[19])
    except (FileNotFoundError,ProcessLookupError):return None

def gpu():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def deadline():return json.loads((CONTROL/'artifacts/deadline.json').read_text())
def admission(stage,hours):
    req=dict(stage=stage,requested_unix=time.time(),required_seconds=hours*3600)
    atomic(Q/'rental_check_request.json',req)
    while True:
        d=deadline()
        if time.time()+hours*3600>=d['unix']:
            status('capacity_pending',stage=stage,required_seconds=hours*3600,cutoff_unix=d['unix'])
        else:
            p=ROOT/'artifacts/platform_verification.json';v=json.loads(p.read_text()) if p.exists() else {}
            if v.get('checked_unix',0)>req['requested_unix'] and v.get('rental_expiry_unix')==d['rental_expiry_unix'] and v.get('cutoff_unix')==d['unix']:
                if shutil.disk_usage(ROOT).free>=16*1024**3:return
                status('capacity_pending',stage=stage,reason='Output disk below 16 GiB')
            else:status('waiting_platform_refresh',stage=stage,request=req)
        time.sleep(20)
def validate():
    p=json.loads((ROOT/'artifacts/protocol.json').read_text())
    for n,h in {**p['source_sha256'],**p['config_sha256']}.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    c=json.loads((ROOT/'artifacts/cpu_acceptance.json').read_text());assert c['passed'] and c['source_sha256']==p['source_sha256']
def run(stage,argv,lease,seconds):
    assert not gpu(),'Another GPU process remains; refuse to overlap'
    validate();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',VLLM_USE_V1='0',NEXT_GPU_LEASE_FD=str(lease.fileno()))
    with (Q/(stage+'.log')).open('a') as f:
        child=subprocess.Popen([PY,'-u',*argv],cwd=ROOT,env=env,pass_fds=(lease.fileno(),),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    status('running',stage=stage,pid=child.pid,start_ticks=identity(child.pid),command=argv)
    end=time.time()+seconds
    while child.poll() is None:
        if time.time()>=min(end,deadline()['unix']):
            atomic(Q/(stage+'_timeout.json'),dict(time=time.time(),pid=child.pid,deadline=deadline()))
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=180)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise TimeoutError(stage+' timed out; evidence retained')
        time.sleep(5)
    assert child.returncode==0,(stage,child.returncode)
    # CUDA descendants must also exit before any successor.
    until=time.time()+120
    while gpu() and time.time()<until:time.sleep(2)
    assert not gpu(),'GPU descendants survived completed stage'
def complete(path):
    return path.exists() and json.loads(path.read_text()).get('status')=='complete'
def main():
    with (Q/'scheduler.lock').open('a+') as single:
        fcntl.flock(single,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            p=B3/'runs/queue/status.json';b=json.loads(p.read_text()) if p.exists() else {}
            if b.get('status')=='B3_seed42_complete':break
            status('waiting_B3_complete',b3_status=b);time.sleep(20)
        for m in ('patr','tree_random'):
            assert complete(B3/f'runs/formal/{m}_seed42/status.json')
            summary=json.loads((B3/f'runs/supplemental500/{m}/summary.json').read_text())
            assert summary['n_tasks']==500 and summary['n_trajectories']==2000
        admission('CERT_v2_acceptance_G1',3)
        with (CONTROL/'runs/control_queue/supervisor.lock').open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert not gpu();validate()
            start=time.time()
            for model in ('sft','sft_grpo400','sft_patr400'):
                result=ROOT/f'runs/acceptance/{model}/result.json'
                if not result.exists():
                    if result.parent.exists():result.parent.rename(result.parent.with_name(model+'_incomplete_'+str(int(time.time()))))
                    run('acceptance_'+model,['scripts/acceptance.py','--model',model],lease,max(1,10800-(time.time()-start)))
                if not json.loads(result.read_text())['passed']:
                    status('technical_acceptance_failed',model=model,no_training=True);return
                if not complete(ROOT/f'runs/g1/{model}/status.json'):
                    run('g1_'+model,['scripts/g1.py','--model',model],lease,max(1,10800-(time.time()-start)))
            run('g1_gate',['scripts/g1_gate.py'],lease,300)
            gate=json.loads((ROOT/'runs/g1/gate.json').read_text())
            if not gate['pass_gate']:status('G1_gate_failed',gate=gate,no_training=True);return
        # Release GPU while waiting for a fresh platform/capacity check for both G2 arms.
        admission('CERT_v2_G2_pair',8.25)
        with (CONTROL/'runs/control_queue/supervisor.lock').open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for selector in ('uct','uniform'):
                if selector=='uniform':admission('CERT_v2_G2_uniform',4.05)
                dest=ROOT/f'runs/g2/{selector}/code/runs/formal/patr_seed42'
                if complete(dest/'status.json'):continue
                cmd=['run_cert.py','--selector',selector]
                if (dest/'checkpoint.pt').exists():cmd+=['--resume']
                run('g2_'+selector,cmd,lease,4*3600)
            run('g2_gate',['scripts/g2_gate.py'],lease,300)
        gate=json.loads((ROOT/'runs/g2/gate.json').read_text())
        status('G2_complete_G3_review_pending' if gate['pass_gate'] else 'G2_gate_failed',gate=gate)
if __name__=='__main__':
    try:main()
    except BlockingIOError:
        raise SystemExit('Queue already locked; no status overwritten and no GPU task started')
    except Exception as e:
        status('error_evidence_retained',error=repr(e));raise
