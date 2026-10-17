import os,json,time,hashlib,subprocess,signal,fcntl,shutil
from pathlib import Path
root=Path(__file__).resolve().parent
py='/home/vipuser/patr-reproduction/.venv/bin/python'
reports=root/'reports';reports.mkdir(exist_ok=True)
lock=(reports/'sequential.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def status(**kw):atomic(reports/'sequential_status.json',dict(time=time.time(),controller_pid=os.getpid(),**kw))
def event(**kw):
 with (reports/'execution_events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**kw))+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def deadline():return json.loads((root/'artifacts/deadline.json').read_text())['unix']
identity={n:sha(root/n) for n in ('train.py','patr_core.py','configs/frozenlake_05b.json','artifacts/data.json','artifacts/runtime_override.json')}
assert (root/'train.py').read_bytes()==Path('/home/vipuser/patr-reproduction/train.py').read_bytes()
for n in ('configs/frozenlake_05b.json','artifacts/data.json','artifacts/runtime_override.json'):assert (root/n).read_bytes()==(Path('/home/vipuser/patr-reproduction')/n).read_bytes()
atomic(reports/'sequential_frozen_source.json',identity)
if not (root/'artifacts/deadline.json').exists():atomic(root/'artifacts/deadline.json',dict(unix=1792183500,reason='User existing Oct17 05:00 Beijing deadline; 15 minute save margin'))
event(kind='user_execution_change',order=['GRPO final-model diagnostic only','base PaTR seed42 400','base Tree-Random seed42 400'],diagnostic_result_blocks_training=False,algorithm_changed=False,old_gate_bypassed_by_explicit_new_authorization=True)
def run(label,cmd):
 assert all(sha(root/n)==h for n,h in identity.items())
 if time.time()>=deadline():status(phase='deadline_saved',next_phase=label);raise SystemExit
 with (reports/(label+'.log')).open('x') as log:
  p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  status(phase=label,pid=p.pid);event(kind='start',phase=label,pid=p.pid)
  signaled=False
  while p.poll() is None:
   if time.time()>=deadline() and not signaled:
    p.send_signal(signal.SIGTERM);signaled=True;event(kind='deadline_sigterm',phase=label,pid=p.pid)
   time.sleep(5)
  event(kind='exit',phase=label,returncode=p.returncode)
  if p.returncode:
   status(phase='serious_execution_failure',failed_phase=label,returncode=p.returncode);raise SystemExit
run('grpo_diagnostic',[py,'-u','grpo_diagnostic.py'])
assert (reports/'grpo_diagnostic/summary.json').exists()
for method in ('patr','tree_random'):
 out=root/'runs/formal'/f'{method}_seed42';out.mkdir(parents=True,exist_ok=True)
 assert not (out/'status.json').exists(), 'Never overwrite an existing experiment'
 # One existing checkpoint + its atomic replacement + final export and margin.
 if shutil.disk_usage(out).free<15*1024**3:
  status(phase='disk_capacity_blocked',method=method,free_bytes=shutil.disk_usage(out).free);raise SystemExit
 run(method+'400',[py,'-u','train.py','--method',method])
 s=json.loads((out/'status.json').read_text())
 if s.get('status')!='complete' or s.get('step')!=400:
  status(phase='saved_incomplete',method=method,run_status=s);raise SystemExit
 assert (out/'final_test.json').exists() and (out/'final_model/model.safetensors').exists()
 atomic(reports/(method+'_completion.json'),dict(run_status=s,checkpoint_sha256=sha(out/'checkpoint.pt'),final_model_sha256=sha(out/'final_model/model.safetensors'),source=identity))
status(phase='both_methods_complete',methods=['patr','tree_random'],next_action='analysis_and_requested_evaluation_only_no_further_training')
