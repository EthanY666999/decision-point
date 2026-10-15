"""One-shot authorized direct400 orchestration; never retries a failed/guarded stage."""
import hashlib,json,os,shutil,subprocess,time,sys
from pathlib import Path
ROOT=Path('/home/vipuser/patr-reproduction-v43-alerts')
sys.path.insert(0,str(ROOT))
PYTHON='/home/vipuser/patr-reproduction/.venv/bin/python'
R1='/home/vipuser/patr-reproduction-v41-r1b/runs/r1b-a/summary.json'
DEADLINE=1792184400
REPORT=ROOT/'reports/v43_pipeline_status.json'
HISTORY_BYTES=1260426553*24
cursors={};streaks={}

def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

def monitor(out):
 from alerts_v43 import advance
 global streaks
 for p in sorted(out.glob('attempt-*/metrics.jsonl')):
  offset=cursors.get(str(p),0)
  with p.open('rb') as f:
   f.seek(offset)
   for line in f:
    if not line.endswith(b'\n'):break
    d=json.loads(line);offset+=len(line)
    for warning in advance(streaks,d):
     warning.update(unix=time.time(),run=str(out))
     with (ROOT/'reports/training_alerts.jsonl').open('a') as stream:stream.write(json.dumps(warning)+'\n')
  cursors[str(p)]=offset
 write(ROOT/'reports/monitor_cursor.json',dict(cursors=cursors,streaks=streaks))

def stage(method,out,limit,resume=False):
 if time.time()>=DEADLINE-900:raise RuntimeError('Deadline before stage')
 cmd=[PYTHON,'-u','train_v4.py','--phase',method,'--out',str(out),'--r1-summary',R1,'--stop-at',str(limit)]
 if resume:cmd.append('--resume')
 history=Path('/data/patr-storage/v43-history')/out.name
 if not resume:
  if out.exists():raise RuntimeError('Refusing existing new run')
  if shutil.disk_usage('/data').free<HISTORY_BYTES+4*1024**3:raise RuntimeError('Insufficient historical-weight capacity')
  history.mkdir(parents=True,exist_ok=False)
 with (ROOT/f'reports/{method}-{limit}.log').open('x') as log:
  proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write(REPORT,dict(state='running',method=method,target=limit,pid=proc.pid,run=str(out),updated=time.time()))
 if not resume:
  # Establish storage before model initialization and first validation snapshot.
  until=time.time()+60
  while not out.exists() and proc.poll() is None and time.time()<until:time.sleep(.1)
  if not out.exists() or (out/'history').exists():
   if proc.poll() is None:proc.terminate()
   raise RuntimeError('Could not establish history storage before training')
  (out/'history').symlink_to(history,target_is_directory=True)
 while proc.poll() is None:
  monitor(out);time.sleep(10)
 monitor(out)
 state=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {'state':'missing_status'}
 if proc.returncode or state.get('state')!='complete' or state.get('counts',{}).get('accepted_iteration')!=limit:
  write(REPORT,dict(state='stopped_requires_review',method=method,target=limit,returncode=proc.returncode,training_status=state,run=str(out),updated=time.time()))
  return False
 return True

def main():
 out=ROOT/'runs/patr-400-alerts-a'
 if not stage('patr',out,400,resume=True):return
 assert (out/'evaluations/reference_test_400.json').exists()
 write(REPORT,dict(state='completed_patr',updated=time.time(),grpo='not_launched_pending_storage_and_time_review'))

if __name__=='__main__':
 try:main()
 except Exception as exc:
  write(REPORT,dict(state='controller_error_requires_review',error=repr(exc),updated=time.time()))
  raise
