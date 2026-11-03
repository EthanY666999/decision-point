"""Small local relay. Complete artifacts live remotely; one response cached locally."""
import fcntl,json,os,pathlib,shutil,subprocess,time,traceback
import generate as g
ROOT=g.ROOT
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','galaxy-4090']
REMOTE=g.REMOTE

def remote(code,payload=None,timeout=45):
    import shlex
    p=subprocess.run(SSH+[REMOTE+'/.venv/bin/python -c '+shlex.quote(code)],input=payload,text=True,capture_output=True,timeout=timeout)
    if p.returncode: raise RuntimeError(p.stderr[-1500:] or p.stdout[-1500:])
    return p.stdout

def main():
    lock=(ROOT/'bridge.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'teacher_workspace').mkdir(exist_ok=True);g.save(ROOT/'response_schema.json',g.SCHEMA)
    awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
    failures=0
    try:
        while time.time()<g.DEADLINE:
            try:
                raw=remote("from pathlib import Path; p=Path('"+REMOTE+"/teacher_pending.json'); print(p.read_text() if p.exists() else '{}')")
                req=json.loads(raw)
                if not req:
                    g.save(ROOT/'bridge_status.json',{'state':'waiting_for_remote','pid':os.getpid(),'updated':time.time()})
                    time.sleep(5);continue
                idx=int(req['index']);turn=int(req['turn']);assert 0<=idx<10000 and 1<=turn<=40
                folder=ROOT/'spool'/f'{idx:04d}';folder.mkdir(parents=True,exist_ok=True)
                g.save(ROOT/'bridge_status.json',{'state':'teacher_call','index':idx,'turn':turn,'pid':os.getpid(),'updated':time.time()})
                response,meta=g.teacher(req['messages'],folder,turn)
                assert meta['prompt_sha256']==req['prompt_sha256']
                # Upload to the remote episode before acknowledgement. No credential files.
                p=subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',str(folder)+'/', 'galaxy-4090:'+REMOTE+f'/episodes/{idx:04d}/'],capture_output=True,text=True,timeout=90)
                if p.returncode:raise RuntimeError('Artifact upload failed: '+p.stderr[-500:])
                remote("import json,hashlib,pathlib; r=pathlib.Path('"+REMOTE+f"/episodes/{idx:04d}'); p=r/'teacher-{turn:03d}'; m=json.loads(p.with_suffix('.meta.json').read_text()); assert m['response_sha256']==hashlib.sha256(p.with_suffix('.response.json').read_bytes()).hexdigest(); t=p.with_suffix('.ready.tmp'); t.write_text(json.dumps({{'uploaded':True}})); t.replace(p.with_suffix('.ready.json'))")
                shutil.rmtree(folder)
                failures=0
                g.save(ROOT/'bridge_status.json',{'state':'delivered','index':idx,'turn':turn,'pid':os.getpid(),'updated':time.time()})
                # Avoid dispatching the same completed mailbox before remote acknowledgement.
                while True:
                    nxt=json.loads(remote("from pathlib import Path; p=Path('"+REMOTE+"/teacher_pending.json'); print(p.read_text() if p.exists() else '{}')"))
                    if not nxt or (nxt['index'],nxt['turn'])!=(idx,turn):break
                    time.sleep(2)
            except (subprocess.TimeoutExpired,RuntimeError,OSError) as exc:
                failures+=1
                g.save(ROOT/'bridge_status.json',{'state':'connection_retry','error':str(exc),'consecutive_errors':failures,'pid':os.getpid(),'updated':time.time()})
                print(str(exc),flush=True)
                if failures>=5:raise
                time.sleep(10)
    except BaseException as exc:
        g.save(ROOT/'bridge_status.json',{'state':'attention_required','error':str(exc),'pid':os.getpid(),'updated':time.time()});raise
    finally:awake.terminate()
if __name__=='__main__':main()
