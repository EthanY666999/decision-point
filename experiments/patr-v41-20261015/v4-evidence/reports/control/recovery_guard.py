"""One-shot recovery reservation, invoked by existing heartbeat after diagnosis.
Never starts a retry loop or changes the training snapshot. Reserve before launch.
"""
import argparse,fcntl,hashlib,json,os,time
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def next_state(state,kind,evidence,now):
 if kind not in ('ram','cuda','disk'):raise ValueError('Unrecognized resource failure')
 if state['total']>=2:raise ValueError('Run recovery budget exhausted')
 key={'ram':'ram_original_restart','cuda':'cuda_original_restart'}.get(kind)
 if key and state[key]>=1:raise ValueError('Original configuration restart exhausted')
 if not evidence.get('cause') or not evidence.get('corrective_action'):raise ValueError('Missing diagnosis')
 if not evidence.get('resource_improvement_verified') or not evidence.get('capacity_sufficient'):raise ValueError('No verified capacity improvement')
 if now>=evidence['deadline_unix']-900:raise ValueError('Deadline reserve reached')
 result=json.loads(json.dumps(state));result['total']+=1
 if key:result[key]+=1
 result['history'].append(dict(kind=kind,time=now,evidence=evidence,state='reserved_before_launch'))
 return result

def reserve(run,kind,evidence_path):
 run=Path(run);evidence=json.loads(Path(evidence_path).read_text())
 if evidence['run']!=str(run.resolve()):raise ValueError('Evidence run mismatch')
 pid=evidence['failed_pid']
 if Path('/proc',str(pid)).exists():raise ValueError('Failed PID still exists; inspect before recovery')
 cp=run/'latest.pt';manifest=json.loads(Path(str(cp)+'.manifest.json').read_text())
 if cp.stat().st_size!=manifest['bytes'] or digest(cp)!=manifest['sha256']:raise ValueError('Latest checkpoint not verified; inspect fallback explicitly')
 if os.statvfs(run).f_bavail*os.statvfs(run).f_frsize<20*1024**3:raise ValueError('Insufficient free disk')
 evidence=dict(evidence,evidence_sha256=digest(Path(evidence_path)),checkpoint_sha256=manifest['sha256'])
 with (run/'recovery.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  p=run/'resource_recoveries.json';state=json.loads(p.read_text())
  state=next_state(state,kind,evidence,time.time());tmp=p.with_suffix('.tmp')
  with tmp.open('w') as f:json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)
 return state

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--kind',choices=['ram','cuda','disk'],required=True);p.add_argument('--evidence',required=True);a=p.parse_args()
 print(json.dumps(reserve(a.run,a.kind,a.evidence)))
