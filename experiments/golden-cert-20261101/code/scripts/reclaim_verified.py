import json,pathlib,hashlib,os,subprocess,shutil,time,fcntl
root=pathlib.Path('/data/patr-storage/cert-space-20261101');root.mkdir(parents=True,exist_ok=True)
lock=(root/'reclaim.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
manifest=json.loads((root/'manifest.json').read_text());receipt={'started':time.time(),'before':{p:dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in ['/','/data']},'files':[]}
def save():
 t=root/'receipt.tmp';t.write_text(json.dumps(receipt,indent=2));os.replace(t,root/'receipt.json')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def users(p):
 found=[];actual=str(p.resolve())
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   for fd in (proc/'fd').iterdir():
    try:
     if os.path.realpath(fd)==actual:found.append(proc.name)
    except OSError:pass
   if actual in (proc/'maps').read_text():found.append(proc.name)
  except (OSError,PermissionError):pass
 return sorted(set(found))
for e in manifest['files']:
 s=pathlib.Path(e['source']);state=json.loads((pathlib.Path(e['run'])/'status.json').read_text())
 assert state['status']==e['expected_status'] and state['step']==state['total_steps']==e['expected_steps']
 assert '/b3/code/runs/formal/' not in str(s)
 assert s.is_file() and not s.is_symlink() and s.stat().st_size==e['bytes']
 assert digest(s)==e['sha256'] and not users(s),str(s)
save()
try:
 for e in sorted(manifest['files'],key=lambda x:x['action']=='migrate'):
  s=pathlib.Path(e['source']);before=s.stat();assert not users(s)
  if e['action']=='migrate':
   d=pathlib.Path(e['destination']);d.parent.mkdir(parents=True,exist_ok=True);assert not d.exists()
   assert shutil.disk_usage('/data').free>e['bytes']+16*1024**3
   partial=d.with_name(d.name+'.copying');subprocess.run(['rsync','-a','--bwlimit=150000',str(s),str(partial)],check=True)
   assert partial.stat().st_size==e['bytes'] and digest(partial)==e['sha256']
   with partial.open('rb') as f:os.fsync(f.fileno())
   assert not users(s) and s.stat().st_mtime_ns==before.st_mtime_ns
   os.replace(partial,d);link=s.with_name(s.name+'.data-link');assert not link.exists();link.symlink_to(d);os.replace(link,s)
  else:
   # Only the verified redundant remote replica is removed; full local bytes retained.
   assert s.stat().st_mtime_ns==before.st_mtime_ns and not users(s)
   s.unlink()
  receipt['files'].append(dict(e,finished=time.time()));save();print(e['action'],str(s),flush=True)
 receipt.update(status='complete_verified',finished=time.time(),after={p:dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in ['/','/data']});save()
except BaseException as ex:
 receipt.update(status='error',error=repr(ex));save();raise
