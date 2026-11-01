"""One-shot verified cleanup of completed B2 PaTR smoke, then original queue recovery."""
import json,hashlib,time,subprocess
from pathlib import Path
BASE=Path('/Users/ethan/Documents/ChatGPT/ACL2027/output/golden-next-20261031')
OUT=BASE/'monitor_20261101_1410';BACK=BASE/'weights-backup-b2_patr_smoke';LABEL='b2_patr_smoke'
while time.time()<1793592584-27000:
 s=json.loads((BACK/'status.json').read_text())
 if s['status']=='complete_verified':break
 if s['status']=='needs_recovery':raise RuntimeError(s)
 time.sleep(10)
else:raise RuntimeError('Insufficient deadline window')
manifest=json.loads((BACK/LABEL/'remote_manifest.json').read_text());receipt=json.loads((BACK/'receipt.json').read_text())
assert len(manifest)==10 and s['files']==10
for name,v in manifest.items():
 f=BACK/LABEL/name;assert not f.is_symlink() and f.stat().st_size==v['bytes'] and receipt[LABEL+'/'+name]['verified']
 h=hashlib.sha256()
 with f.open('rb') as src:
  for b in iter(lambda:src.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==v['sha256']
(OUT/'b2_smoke_local_reverification.json').write_text(json.dumps(dict(time=time.time(),files=manifest,all_match=True),indent=2)+'\n')
code='''import json,pathlib,time,hashlib,subprocess,fcntl,os,shutil
M=json.loads(MANIFEST)
root=pathlib.Path('/home/vipuser/golden-next-20261031/b2/code');r=root/'runs/smoke/patr_seed42';q=root/'runs/queue'
state=json.loads((q/'status.json').read_text());assert state['status']=='capacity_pending' and state['method']=='patr' and not state['smoke'],state
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\\0')
  assert not any(x.endswith(b'b2_queue.py') or x.endswith(b'run_b2.py') for x in args),(p.name,args)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
lease=(pathlib.Path('/home/vipuser/golden-controls-20261031/code/runs/control_queue/supervisor.lock')).open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
own=(q/'scheduler.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
s=json.loads((r/'status.json').read_text());assert s['status']=='smoke_complete' and s['step']==s['total_steps']==1
assert json.loads((r/'technical_acceptance.json').read_text())['passed']
actual={str(p.relative_to(r)) for p in (r/'final_model').rglob('*') if p.is_file()}|{'checkpoint.pt'};assert actual==set(M)
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  assert str(r) not in (p/'maps').read_text()
  for fd in (p/'fd').iterdir():
   try:assert not os.readlink(fd).startswith(str(r)+'/')
   except OSError:pass
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
for name,v in M.items():
 f=r/name;assert not f.is_symlink() and f.stat().st_size==v['bytes']
 h=hashlib.sha256()
 with f.open('rb') as src:
  for b in iter(lambda:src.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==v['sha256']
proof=dict(time=time.time(),root=str(r),files=M,all_match=True,prior_status=state)
(q/'capacity_recovery_20261101_1410_verified.json').write_text(json.dumps(proof,indent=2)+'\\n')
for name in M:(r/name).unlink()
(r/'final_model').rmdir()
proof.update(removed_bytes=sum(v['bytes'] for v in M.values()),free_bytes=shutil.disk_usage(root).free)
(r/'verified_redundant_weights_cleanup_20261101_1410.json').write_text(json.dumps(proof,indent=2)+'\\n')
assert proof['free_bytes']>=16*1024**3
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
fcntl.flock(own,fcntl.LOCK_UN);own.close();fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
with (q/'supervisor.log').open('a') as log:
 p=subprocess.Popen(['/home/vipuser/patr-reproduction/.venv/bin/python','-u','scripts/b2_queue.py'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
proof.update(restarted_supervisor_pid=p.pid,restarted_unix=time.time())
(q/'capacity_recovery_20261101_1410.json').write_text(json.dumps(proof,indent=2)+'\\n')
print(json.dumps(proof,indent=2))
'''.replace('MANIFEST',repr(json.dumps(manifest)))
x=subprocess.run(['ssh','galaxy-4090','/home/vipuser/patr-reproduction/.venv/bin/python -'],input=code,text=True,capture_output=True)
if x.returncode:
 (OUT/'capacity_recovery_error.txt').write_text(x.stderr);raise RuntimeError(x.stderr)
(OUT/'capacity_recovery_receipt.json').write_text(x.stdout);(BACK/'remote_cleanup_receipt.json').write_text(x.stdout)
print('Recovery complete',json.loads(x.stdout)['restarted_supervisor_pid'],flush=True)
