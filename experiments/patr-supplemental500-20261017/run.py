import subprocess,time,os,json,fcntl,hashlib
from pathlib import Path
root=Path('/home/vipuser/patr-v1-tree-20261016');out=root/'reports/eval500-20261017';py='/home/vipuser/patr-reproduction/.venv/bin/python'
def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (out/'protocol.json').exists()
assert not (out/'controller_status.json').exists(),'Do not repeat this evaluation'
identity={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['protocol.json','tasks.json','maps.json','eval500.py','run.py']}
atomic(out/'execution_identity.json',identity)
for method in ['grpo','patr','tree_random']:
 assert all(hashlib.sha256((out/n).read_bytes()).hexdigest()==h for n,h in identity.items())
 deadline=json.loads((root/'artifacts/deadline.json').read_text())['unix'];assert time.time()<deadline
 with (out/f'{method}.log').open('x') as log:
  p=subprocess.Popen([py,'-u',str(out/'eval500.py'),method],cwd=root,stdout=log,stderr=subprocess.STDOUT)
  atomic(out/'controller_status.json',dict(phase=method,controller_pid=os.getpid(),evaluation_pid=p.pid,time=time.time(),training=False))
  try:r=p.wait(timeout=min(3600,deadline-time.time()))
  except subprocess.TimeoutExpired:
   p.terminate();atomic(out/'controller_status.json',dict(phase='failed_timeout',method=method,time=time.time()));raise SystemExit(1)
 if r:
  atomic(out/'controller_status.json',dict(phase='failed',method=method,returncode=r,time=time.time()));raise SystemExit(r)
atomic(out/'controller_status.json',dict(phase='all_three_evaluations_complete',time=time.time(),training=False))
