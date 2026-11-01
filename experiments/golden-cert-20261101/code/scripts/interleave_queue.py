"""CPU handover: preserve the exact live PaTR arm, then supplemental -> G1 -> Random.

Never signal the adopted training PID. It inherited the old exclusive GPU lease.
The CPU supervisor replacement is recorded in handover.json before activation.
"""
import fcntl,hashlib,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
B3=Path('/home/vipuser/golden-next-20261031/b3/code')
C=Path('/home/vipuser/golden-controls-20261031/code')
PY='/home/vipuser/patr-reproduction/.venv/bin/python'
OUT=ROOT/'runs/queue'
def read(p):return json.loads(p.read_text())
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def owners():return {int(x) for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()}
def identity(pid):
 try:
  s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
  return None if s[0]=='Z' else int(s[19])
 except FileNotFoundError:return None
def deadline():return read(C/'artifacts/deadline.json')['unix']
def verified_lease():
 d=read(C/'artifacts/deadline.json');v=read(ROOT/'artifacts/platform_verification.json')
 return (v['rental_expiry_unix']==d['rental_expiry_unix'] and v['cutoff_unix']==d['unix'] and time.time()-v['checked_unix']<=24*3600)
def wait_platform(stage):
 requested=time.time()
 save(OUT/'rental_check_request.json',dict(stage=stage,requested_unix=requested,url='https://gpu.ai-galaxy.com/console/dashboard',instruction='Heartbeat must refresh the platform, then save matching observed expiry/cutoff and checked_unix in artifacts/platform_verification.json; never infer renewal.'))
 while time.time()<deadline():
  v=read(ROOT/'artifacts/platform_verification.json')
  if v.get('checked_unix',0)>=requested and verified_lease():return True
  save(OUT/'status.json',dict(status='waiting_platform_refresh',stage=stage,pid=os.getpid(),requested_unix=requested,gpu_initialized=False))
  time.sleep(15)
 return False

def ready():
 p=read(ROOT/'artifacts/protocol.json');a=read(ROOT/'artifacts/cpu_acceptance.json')
 return a['passed'] and a['source_sha256']==p['source_sha256'] and all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in {**p['source_sha256'],**p['config_sha256']}.items())
def children(pid):
 result={pid}
 for _ in range(8):
  for p in Path('/proc').iterdir():
   if p.name.isdigit():
    try:
     if int((p/'stat').read_text().rsplit(')',1)[1].split()[1]) in result:result.add(int(p.name))
    except (OSError,ValueError):pass
 return result

def run(command,cwd,name,lease,until):
 assert not owners()
 with (OUT/(name+'.log')).open('a') as log:
  child=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(lease.fileno(),),env=dict(os.environ,NEXT_GPU_LEASE_FD=str(lease.fileno())))
  save(OUT/'status.json',dict(status='running',stage=name,pid=child.pid,supervisor_pid=os.getpid(),started=time.time(),command=command))
  sent=False
  while child.poll() is None:
   if time.time()>=min(deadline(),until) or owners()-children(child.pid):
    if not sent:child.send_signal(signal.SIGTERM);sent=True
   time.sleep(5)
 return child.returncode

def main():
 OUT.mkdir(parents=True,exist_ok=True);own=(OUT/'scheduler.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 hand=read(OUT/'handover.json');adopt=hand['trainer_pid'];ticks=hand['trainer_start_ticks']
 save(OUT/'status.json',dict(status='waiting_live_B3_PaTR',pid=os.getpid(),adopted_pid=adopt,adopted_start_ticks=ticks,gpu_initialized=False))
 while identity(adopt)==ticks:time.sleep(15)
 # A missing checkpoint is not a failure for a previously backed-up completed run.
 s=read(B3/'runs/formal/patr_seed42/status.json')
 if s.get('status')!='complete' or s.get('step')!=400:
  save(OUT/'status.json',dict(status='needs_recovery',reason='Adopted B3 arm not complete',arm_status=s));return
 lease=(C/'runs/control_queue/supervisor.lock').open('a')
 fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not owners()
 assert ready(),'CERT source acceptance unavailable'
 # Reserve Random400 six hours, smoke/supp fifteen minutes each, margin fifteen.
 reserve=21600+2700
 supp=B3/'runs/supplemental500/patr'
 if not (supp/'summary.json').exists():
  if deadline()-time.time()<reserve+900:
   save(OUT/'status.json',dict(status='capacity_pending',stage='patr_supplemental'));return
  if supp.exists():supp.rename(supp.with_name('patr_incomplete_'+str(int(time.time()))))
  rc=run([PY,'-u','scripts/golden_supplemental.py','--model',str(B3/'runs/formal/patr_seed42/final_model'),'--output',str(supp)],B3,'B3_patr_supplemental',lease,deadline())
  if rc or not (supp/'summary.json').exists():
   save(OUT/'status.json',dict(status='needs_recovery',stage='B3_patr_supplemental',returncode=rc));return
 if not wait_platform('CERT_G1'):return
 if deadline()-time.time()<reserve+7200:
  save(OUT/'status.json',dict(status='capacity_pending',reason='Full two-hour CERT G1 plus remaining B3 budget unavailable; no silent shortening'));return
 if shutil.disk_usage(ROOT).free<16*1024**3:
  save(OUT/'status.json',dict(status='capacity_pending',reason='Data disk below 16GiB'));return
 g1_end=time.time()+7200;failed=False
 for model in ('sft','sft_grpo400','sft_patr400'):
  status=ROOT/'runs/g1'/model/'status.json'
  if status.exists() and read(status).get('status')=='complete':continue
  rc=run([PY,'-u','scripts/g1.py','--model',model],ROOT,'G1_'+model,lease,g1_end)
  if rc:
   failed=True;save(OUT/'cert_failure.json',dict(model=model,returncode=rc,time=time.time(),policy='Stop CERT, preserve evidence, continue original Random arm'));break
 if not failed:
  r=subprocess.run([PY,'scripts/g1_gate.py'],cwd=ROOT,capture_output=True,text=True)
  (OUT/'g1_gate.log').write_text(r.stdout+r.stderr)
  if r.returncode:save(OUT/'cert_failure.json',dict(stage='gate',returncode=r.returncode,time=time.time()))
 assert not owners()
 if not wait_platform('B3_Random400'):return
 fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
 # Original frozen B3 queue skips completed PaTR/supp and resumes Random sequence.
 with (OUT/'B3_remaining.log').open('a') as log:
  child=subprocess.Popen([PY,'-u','scripts/b3_queue.py'],cwd=B3,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  save(OUT/'status.json',dict(status='B3_remaining',supervisor_pid=os.getpid(),b3_supervisor_pid=child.pid))
  rc=child.wait()
 s=read(B3/'runs/queue/status.json')
 save(OUT/'status.json',dict(status='interleave_complete' if s.get('status')=='B3_seed42_complete' else 'needs_recovery',b3_status=s,returncode=rc,finished=time.time(),next='Read G1 gate; G2 requires full B3 completion, platform refresh, capacity and its own preflight. No automatic G2 from this waiter.'))
if __name__=='__main__':main()
