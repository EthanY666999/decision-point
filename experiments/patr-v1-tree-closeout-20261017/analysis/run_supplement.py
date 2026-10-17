import subprocess,json,time,fcntl,os
from pathlib import Path
out=Path('/home/vipuser/patr-v1-tree-20261016/reports/closeout-20261017')
lock=(out/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
py='/home/vipuser/patr-reproduction/.venv/bin/python'
for method in ['grpo','patr','tree_random']:
 assert time.time()<1792284300
 assert not (out/f'{method}_map8.json').exists()
 (out/'evaluation_status.json').write_text(json.dumps(dict(phase=method,pid=os.getpid(),time=time.time())))
 with (out/f'{method}_map8.log').open('x') as log:
  result=subprocess.run([py,str(out/'supplement.py'),method],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 if result.returncode:
  (out/'evaluation_status.json').write_text(json.dumps(dict(phase='failed',method=method,returncode=result.returncode,time=time.time())))
  raise SystemExit(result.returncode)
(out/'evaluation_status.json').write_text(json.dumps(dict(phase='complete',time=time.time(),methods=['grpo','patr','tree_random'])))
