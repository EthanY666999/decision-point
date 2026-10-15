"""User-approved bounded execution recovery; never changes learning settings."""
import os,sys,json,time,subprocess,signal,fcntl,hashlib,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/patr-400-xformers-a'; REPORT=ROOT/'reports'
PY='/home/vipuser/patr-reproduction/.venv/bin/python'; DEADLINE=1792184400
STATE=REPORT/'recovery_state.json'; EVENTS=REPORT/'fault_handling.jsonl'; STATUS=REPORT/'v43_pipeline_status.json'
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default or {}
def write(p,d):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p)
def event(**d):
 d.update(time=time.time(),user_authorized_execution_change=True)
 with EVENTS.open('a') as f:f.write(json.dumps(d)+'\n');f.flush();os.fsync(f.fileno())
def classify(text):
 t=text.lower()
 if any(x in t for x in ['out of memory','cuda error: out of memory']):return 'oom'
 if any(x in t for x in ['nonfinite','non-finite','floatingpointerror','nan gradient','infinite gradient']):return 'nonfinite'
 if any(x in t for x in ['no space left','insufficient space','disk history reserve','capacity budget','below 12gib','less than 12gib']):return 'disk'
 if any(x in t for x in ['illegal memory access','misaligned address','device-side assert']):return 'cuda_memory'
 if 'cuda error' in t or 'cublas' in t:return 'cuda_other'
 return 'unknown'
def decide(s,kind,batch):
 s['last_failure_streak']=s.get('last_failure_streak',0)+1 if s.get('last_failed_batch')==batch else 1
 s['last_failed_batch']=batch
 if kind=='nonfinite':s.setdefault('nan_batches',[]).append(batch)
 if s['last_failure_streak']>=2:return False,'same batch failed twice'
 if s.get('total',0)>=5:return False,'five automatic recoveries exhausted'
 if kind=='nonfinite' and sum(0<=batch-b<20 for b in s['nan_batches'])>=3:return False,'three nonfinite failures within 20 batches'
 if kind=='nonfinite' and s['nan_batches'].count(batch)>1:return False,'nonfinite retry failed'
 if kind.startswith('cuda') and s.get('cuda_retries',{}).get(kind,0)>=2:return False,'two same-class CUDA retries exhausted'
 if kind=='oom' and s.get('memory_fraction',.4)==.35:return False,'authorized memory reduction already used'
 if kind=='unknown':return False,'unclassified error requires review'
 return True,'authorized bounded recovery'
def verify(cp):
 sys.path.insert(0,str(ROOT));import torch;torch.set_num_threads(4)
 from checkpoint_v4 import verified_load
 st=verified_load(cp)
 assert st['fingerprint']==read(OUT/'fingerprint.json'),'checkpoint identity mismatch'
 def finite(x):
  if isinstance(x,torch.Tensor):return not x.is_floating_point() or bool(torch.isfinite(x).all())
  if isinstance(x,dict):return all(finite(v) for v in x.values())
  if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
  return True
 assert finite(st['model']) and finite(st['optimizer']),'nonfinite checkpoint'
 return st['counters']
def preserve(cp,label):
 d=OUT/'recovery_checkpoints';d.mkdir(exist_ok=True)
 dest=d/(label+'.pt')
 if not dest.exists():os.link(cp,dest);shutil.copy2(str(cp)+'.manifest.json',str(dest)+'.manifest.json')
 return str(dest)
def cleanup_disk():
 # Only current-run superseded previous and pip's rebuildable download cache.
 latest=OUT/'latest.pt';old=OUT/'latest.pt.previous'
 lc=verify(latest)
 if old.exists():
  oc=verify(old)
  if lc['accepted_iteration']>oc['accepted_iteration']:
   old.unlink();Path(str(old)+'.manifest.json').unlink();event(action='delete_verified_superseded_previous',checkpoint=str(old),result='deleted')
 cache=Path.home()/'.cache/pip'
 if cache.exists():shutil.rmtree(cache);event(action='delete_pip_cache',result='deleted')
 return shutil.disk_usage(OUT).free>=12*1024**3
cursors={};streaks={}
def monitor():
 sys.path.insert(0,str(ROOT));from alerts_v43 import advance
 for p in sorted(OUT.glob('attempt-*/metrics.jsonl')):
  pos=cursors.get(str(p),0)
  with p.open('rb') as f:
   f.seek(pos)
   for line in f:
    if not line.endswith(b'\n'):break
    d=json.loads(line);pos+=len(line)
    for warning in advance(streaks,d):event(action='report_only',batch=d['step'],warning=warning,result='training continues')
  cursors[str(p)]=pos
 write(REPORT/'monitor_cursor.json',dict(cursors=cursors,streaks=streaks))
def main():
 lock=(REPORT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 s=read(STATE,dict(total=0,memory_fraction=.4,cuda_retries={},nan_batches=[]));write(STATE,s)
 oldmonitor=read(REPORT/'monitor_cursor.json');cursors.update(oldmonitor.get('cursors',{}));streaks.update(oldmonitor.get('streaks',{}))
 while True:
  if time.time()>=DEADLINE-900:write(STATUS,dict(state='deadline_safe_stop',updated=time.time()));return
  counts=verify(OUT/'latest.pt')
  stamp=time.time_ns();logpath=REPORT/f'training-{stamp}.log'
  env=dict(os.environ,PATR_EXEC_MEMORY_FRACTION=str(s['memory_fraction']))
  cmd=[PY,'-u','train_v4.py','--phase','patr','--out',str(OUT),'--r1-summary','/home/vipuser/patr-reproduction-v41-r1b/runs/r1b-a/summary.json','--resume','--stop-at','400']
  with logpath.open('x') as log:proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(STATUS,dict(state='running',pid=proc.pid,controller_pid=os.getpid(),run=str(OUT),log=str(logpath),checkpoint_counts=counts,recoveries=s['total'],memory_fraction=s['memory_fraction'],updated=time.time()))
  event(action='launch',batch=counts['raw_batch_index']+1,result='process started, not proof of update',pid=proc.pid,log=str(logpath),recoveries=s['total'])
  deadline_signal=False
  while proc.poll() is None:
   monitor()
   if time.time()>=DEADLINE-900 and not deadline_signal:proc.send_signal(signal.SIGTERM);deadline_signal=True;event(action='deadline_graceful_save_requested',result='waiting batch boundary')
   if time.time()>=DEADLINE-60:
    os.killpg(proc.pid,signal.SIGKILL);event(action='deadline_forced_stop',result='last verified checkpoint retained; in-flight work not saved');break
   time.sleep(10)
  proc.wait();monitor();status=read(OUT/'status.json')
  if status.get('state')=='complete' and status.get('training_steps_completed')==400:
   write(STATUS,dict(state='completed_patr',training_status=status,recoveries=s,updated=time.time()));event(action='finished',batch=400,result='training and test complete');return
  # Release only descendants belonging to this launch, before any restart.
  try:os.killpg(proc.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  if deadline_signal or time.time()>=DEADLINE-900:
   write(STATUS,dict(state='deadline_safe_stop',training_status=status,updated=time.time()));return
  tail=logpath.read_bytes()[-65536:].decode(errors='replace');kind=classify(json.dumps(status)+'\n'+tail)
  c=status.get('counts_in_memory',status.get('counts',counts));batch=c.get('raw_batch_index',counts['raw_batch_index'])+1
  permitted,reason=decide(s,kind,batch);write(STATE,s)
  event(action='failure',batch=batch,fault=kind,result=reason,returncode=proc.returncode,status=status)
  if not permitted:write(STATUS,dict(state='safe_stop',reason=reason,training_status=status,recoveries=s,updated=time.time()));return
  cp=OUT/'latest.pt'
  try:
   try:verify(cp)
   except Exception as e:
    previous=OUT/'latest.pt.previous';verify(previous)
    quarantine=OUT/f'failed-checkpoint-{stamp}';quarantine.mkdir();
    for p in (cp,Path(str(cp)+'.manifest.json')):
     if p.exists():os.replace(p,quarantine/p.name)
    os.link(previous,cp);shutil.copy2(str(previous)+'.manifest.json',str(cp)+'.manifest.json');event(action='restore_valid_previous',batch=batch,result=str(e))
   saved=preserve(cp,f'recovery-{s["total"]+1}-{stamp}')
   if kind=='disk' and not cleanup_disk():raise OSError('authorized cleanup insufficient')
   if kind=='oom':s['memory_fraction']=.35
   if kind.startswith('cuda'):s['cuda_retries'][kind]=s['cuda_retries'].get(kind,0)+1
   s['total']+=1;write(STATE,s)
   if (OUT/'status.json').exists():shutil.copy2(OUT/'status.json',REPORT/f'failure-status-{stamp}.json')
   write(OUT/'status.json',dict(state='authorized_recovery',reason=reason,source=saved,counts=verify(cp)))
   event(action='recover_from_verified_checkpoint',batch=batch,fault=kind,result='retry scheduled',source=saved,recoveries=s['total'],memory_fraction=s['memory_fraction'])
  except Exception as e:
   event(action='recovery_failed',batch=batch,result=repr(e));write(STATUS,dict(state='safe_stop',reason=repr(e),recoveries=s,updated=time.time()));return
if __name__=='__main__':
 try:main()
 except Exception as exc:
  event(action='controller_error',result=repr(exc));write(STATUS,dict(state='safe_stop_controller_error',error=repr(exc),updated=time.time()));raise
