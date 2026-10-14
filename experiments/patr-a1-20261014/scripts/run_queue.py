"""Detached sequential runner. Does not change method parameters on failures."""
import datetime,json,os,subprocess,sys,time,traceback,hashlib
import fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patr_core import ROOT,atomic_json,append_json


def main():
    lock=(ROOT/'artifacts/queue.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((ROOT/'configs/frozenlake_05b.json').read_text())
    queue=ROOT/'artifacts/queue_status.json'
    start=time.time()
    # A conservative run window for the user's renewed one-day-plus instance.
    deadline_file=ROOT/'artifacts/deadline.json'
    if not deadline_file.exists():
        atomic_json(deadline_file,dict(unix=start+24*3600,reason='24h checkpoint guard for renewed instance; not a cloud shutdown'))
    sources=list(ROOT.glob('*.py'))+list((ROOT/'configs').glob('*.json'))+list((ROOT/'scripts').glob('*.py'))
    sources += [p for p in (ROOT/'vendor').rglob('*') if p.is_file() and (p.suffix=='.py' or p.name=='LICENSE')]
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    atomic_json(ROOT/'artifacts/source_manifest.json',dict(sha256=files,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    errors=[];stop_reason=None;stopped_method=None
    for method in cfg['methods']:
        out=ROOT/'runs/formal'/f"{method}_seed{cfg['seed']}"
        status_path=out/'status.json'
        old=json.loads(status_path.read_text()) if status_path.exists() else {}
        if old.get('status')=='complete':
            if subprocess.run([sys.executable,'scripts/verify_final.py'],cwd=ROOT).returncode:
                stop_reason='final_verification_failed';stopped_method=method;break
            continue
        if time.time()>=json.loads(deadline_file.read_text())['unix']:
            stop_reason='deadline';stopped_method=method;break
        atomic_json(queue,dict(status='running',method=method,pid=os.getpid(),started_at=start,errors=errors))
        cmd=[sys.executable,'-u','train.py','--method',method,'--resume']
        logfile=ROOT/'logs'/f'formal_{method}.log'
        for attempt in range(4):
            with logfile.open('a') as log:
                log.write('\nSTART '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n');log.flush()
                attempt_offset=log.tell()
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    subprocess.run([sys.executable,'scripts/report.py'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    time.sleep(60)
                rc=child.returncode
            if not rc:break
            # Never classify a new failure using an older attempt's OOM message.
            with logfile.open() as log:
                log.seek(attempt_offset)
                tail=log.read()[-16000:].lower()
            # Bounded engineering recovery only; no algorithm/reward/step changes.
            if 'out of memory' not in tail or attempt==3:break
            override_path=ROOT/'artifacts/runtime_override.json'
            runtime=json.loads(override_path.read_text()) if override_path.exists() else {}
            previous=runtime.get('micro_batch_size',cfg['micro_batch_size'])
            runtime['micro_batch_size']=max(1,previous//2)
            runtime['micro_batch_token_limit']=max(2000,runtime.get('micro_batch_token_limit',cfg['micro_batch_token_limit'])//2)
            runtime['vllm_max_sequences']=max(32,runtime.get('vllm_max_sequences',cfg['vllm_max_sequences'])//2)
            atomic_json(override_path,runtime)
            append_json(ROOT/'logs/engineering_recovery.jsonl',dict(method=method,attempt=attempt+1,reason='CUDA OOM',runtime=runtime,time=time.time()))
        if rc:
            error=dict(method=method,returncode=rc,time=time.time())
            errors.append(error);append_json(ROOT/'logs/queue_errors.jsonl',error)
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(status='failed',returncode=rc,log=f'logs/formal_{method}.log')
            atomic_json(status_path,status)
            stop_reason='method_failed';stopped_method=method
            break
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('status')!='complete':
            stop_reason='method_paused_or_incomplete';stopped_method=method
            break
        # Verify the actual final trajectories before allowing the next method.
        verified=subprocess.run([sys.executable,'scripts/verify_final.py'],cwd=ROOT).returncode
        if verified:
            error=dict(method=method,verification_returncode=verified,time=time.time())
            errors.append(error);append_json(ROOT/'logs/queue_errors.jsonl',error)
            stop_reason='final_verification_failed';stopped_method=method
            break
        subprocess.run([sys.executable,'scripts/report.py'],cwd=ROOT)
    complete=[]
    for method in cfg['methods']:
        p=ROOT/'runs/formal'/f"{method}_seed{cfg['seed']}"/'status.json'
        if p.exists() and json.loads(p.read_text()).get('status')=='complete':complete.append(method)
    all_verified=False
    if len(complete)==len(cfg['methods']) and stop_reason is None:
        all_verified=subprocess.run([sys.executable,'scripts/verify_final.py','--require-complete'],cwd=ROOT).returncode==0
        if not all_verified:stop_reason='final_verification_failed'
    atomic_json(queue,dict(status='complete' if all_verified else 'stopped_incomplete',completed=complete,errors=errors,
                          stop_reason=stop_reason,stopped_method=stopped_method,wall_hours=(time.time()-start)/3600))
    subprocess.run([sys.executable,'scripts/report.py'],cwd=ROOT)

if __name__=='__main__':main()
