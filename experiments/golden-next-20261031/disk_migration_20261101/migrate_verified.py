import json,pathlib,hashlib,os,subprocess,shutil,time,fcntl
root=pathlib.Path('/data/patr-storage/golden-migrated-20261101');root.mkdir(parents=True,exist_ok=True)
lock=(root/'migration.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
manifest=json.loads((root/'manifest.json').read_text());receipt={'started':time.time(),'before':{p:dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in ['/','/data']},'files':[]}
def save():
 tmp=root/'receipt.tmp';tmp.write_text(json.dumps(receipt,indent=2));os.replace(tmp,root/'receipt.json')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def users(p):
 found=[]
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   for fd in (proc/'fd').iterdir():
    try:
     if os.path.realpath(fd)==str(p):found.append(proc.name)
    except OSError:pass
   if str(p) in (proc/'maps').read_text():found.append(proc.name)
  except (OSError,PermissionError):pass
 return sorted(set(found))
assert pathlib.Path('/data').stat().st_dev!=pathlib.Path('/').stat().st_dev
assert shutil.disk_usage('/data').free>manifest['bytes']+10*1024**3
for run in {str(pathlib.Path(e['source']).parent.parent) if '/final_model/' in e['source'] else str(pathlib.Path(e['source']).parent) for e in manifest['files']}:
 status=json.loads((pathlib.Path(run)/'status.json').read_text());assert status['status']=='complete',(run,status)
for e in manifest['files']:
 s=pathlib.Path(e['source']);assert not s.is_symlink();assert s.is_file() and s.stat().st_size==e['bytes'];assert not users(s),(s,users(s))
save()
try:
 for i,e in enumerate(manifest['files']):
  s=pathlib.Path(e['source']);d=pathlib.Path(e['destination']);d.parent.mkdir(parents=True,exist_ok=True);assert not d.exists()
  before=s.stat();assert digest(s)==e['sha256'],str(s);assert not users(s)
  partial=d.with_name(d.name+'.copying');subprocess.run(['rsync','-a','--bwlimit=150000',str(s),str(partial)],check=True)
  assert partial.stat().st_size==e['bytes'] and digest(partial)==e['sha256'],str(partial)
  with partial.open('rb') as f:os.fsync(f.fileno())
  assert s.stat().st_size==before.st_size and s.stat().st_mtime_ns==before.st_mtime_ns;assert not users(s)
  os.replace(partial,d)
  link=s.with_name(s.name+'.data-link');assert not link.exists();link.symlink_to(d);os.replace(link,s)
  assert s.is_symlink() and s.resolve()==d and s.stat().st_size==e['bytes']
  receipt['files'].append(dict(e,migrated_unix=time.time(),source_replaced_by_symlink=True));save();print(json.dumps({'done':i+1,'total':len(manifest['files']),'file':str(s),'bytes':e['bytes']}),flush=True)
 receipt['status']='complete_verified';receipt['finished']=time.time();receipt['after']={p:dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in ['/','/data']};save()
except BaseException as ex:
 receipt['status']='error';receipt['error']=repr(ex);save();raise
