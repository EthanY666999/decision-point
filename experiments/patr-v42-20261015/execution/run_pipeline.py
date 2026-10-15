"""One-shot authorized80->400 orchestration; never retries a failed/guarded stage."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/vipuser/patr-reproduction-v42')
PYTHON='/home/vipuser/patr-reproduction/.venv/bin/python'
R1='/home/vipuser/patr-reproduction-v41-r1b/runs/r1b-a/summary.json'
DEADLINE=1792184400
REPORT=ROOT/'reports/v42_pipeline_status.json'
HISTORY_BYTES=1260426553*24
cursors={};streak=0

def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

def monitor(out):
 global streak
 for p in sorted(out.glob('attempt-*/metrics.jsonl')):
  offset=cursors.get(str(p),0)
  with p.open('rb') as f:
   f.seek(offset)
   for line in f:
    if not line.endswith(b'\n'):break
    d=json.loads(line);offset+=len(line)
    streak=streak+1 if d.get('fence_loop_fraction',0)>.01 else 0
    if streak==3:
     warning=dict(kind='fence_loop_3_consecutive',step=d['step'],raw_batch=d['raw_batch'],fraction=d['fence_loop_fraction'],unix=time.time(),run=str(out),action='report_only_training_continues')
     with (ROOT/'reports/fence_loop_alerts.jsonl').open('a') as stream:stream.write(json.dumps(warning)+'\n')
  cursors[str(p)]=offset

def stage(method,out,limit,resume=False):
 if time.time()>=DEADLINE-900:raise RuntimeError('Deadline before stage')
 cmd=[PYTHON,'-u','train_v4.py','--phase',method,'--out',str(out),'--r1-summary',R1,'--stop-at',str(limit)]
 if resume:cmd.append('--resume')
 history=Path('/data/patr-storage/v42-history')/out.name
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
 out=ROOT/'runs/patr-400-v42-a'
 if not stage('patr',out,80):return
 summary=json.loads((out/'summary.json').read_text())
 assert summary['restore_verified'] and summary['training_steps_completed']==80
 for p in out.glob('attempt-*/metrics.jsonl'):
  assert all(not json.loads(line).get('guard_reasons') for line in p.read_text().splitlines())
 manifest=json.loads((out/'latest.pt.manifest.json').read_text())
 with (out/'latest.pt').open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==manifest['sha256']
 for name in ('summary.json','status.json'):shutil.copy2(out/name,ROOT/'reports'/('milestone80-'+name))
 if not stage('patr',out,400,True):return
 assert (out/'evaluations/reference_test_400.json').exists()
 remaining=DEADLINE-time.time()
 if remaining<14*3600:
  write(REPORT,dict(state='completed_patr_grpo_not_started',reason='less_than_14_hours_remaining',remaining_seconds=remaining));return
 if shutil.disk_usage('/data').free<HISTORY_BYTES+4*1024**3:
  write(REPORT,dict(state='completed_patr_grpo_storage_blocked',reason='historical_weight_capacity_insufficient',remaining_seconds=remaining));return
 global streak
 streak=0
 if stage('grpo',ROOT/'runs/grpo-400-v42-a',400):write(REPORT,dict(state='completed_both',updated=time.time()))

if __name__=='__main__':
 try:main()
 except Exception as exc:
  write(REPORT,dict(state='controller_error_requires_review',error=repr(exc),updated=time.time()))
  raise
