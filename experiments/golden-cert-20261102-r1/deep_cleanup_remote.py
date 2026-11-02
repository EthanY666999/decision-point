"""Audit and remove exact redundant server weight copies; preserve restore indexes."""
import pathlib,json,hashlib,os,time,fcntl,shutil,sys
R=pathlib.Path('/home/vipuser/golden-cert-20261102-r1/code/artifacts/deep_disk_20261103')
def save(n,d):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def users(p):
 actual=str(p.resolve());found=[]
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   for fd in (proc/'fd').iterdir():
    if os.path.realpath(fd)==actual:found.append(proc.name)
   if actual in (proc/'maps').read_text():found.append(proc.name)
  except (OSError,PermissionError):pass
 return sorted(set(found))
def protected():
 expected=json.load((R.parent/'disk_20261103/protected_inputs_sha256.json').open());base=pathlib.Path('/home/vipuser/golden-controls-20261031/code')
 actual={k:sha(base/k) for k in expected};assert actual==expected;return actual
def disks():return {p:dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in ['/','/data']}
lock=pathlib.Path('/home/vipuser/golden-controls-20261031/code/runs/control_queue/supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if sys.argv[1]=='audit':
 c=json.load((R/'local_candidates.json').open());assert c['complete']
 ar=json.load((R/'archive_members.json').open());assert ar['complete']
 archive=pathlib.Path(ar['archive']);archive_sha=sha(archive);assert archive_sha=='2a7670bf862c4b53eaf7d15679abfd2febc834ec05ff86bc6f50d9883e474812'
 matches={e['sha256']:e for e in ar['large_members']}
 candidates=list(c['files'])
 for e in json.load((R/'large_files.json').open()):
  if e['sha256'] in matches and '/patr-reproduction/runs/' in e['path']:
   a=matches[e['sha256']];assert e['bytes']==a['bytes']
   candidates.append(dict(source=e['path'],label='historical_v1',bytes=e['bytes'],sha256=e['sha256'],backup_kind='retained_remote_archive',archive=str(archive),archive_sha256=archive_sha,archive_member=a['name']))
 audit={'started':time.time(),'before':disks(),'protected':protected(),'archive_sha256':archive_sha,'files':[],'absent_preserved_receipts':[]}
 seen=set()
 for e in candidates:
  p=pathlib.Path(e['source'])
  if not p.exists():audit['absent_preserved_receipts'].append(e['source']);continue
  t=p.resolve(strict=True)
  if str(t) in seen:continue
  assert t.name in ['checkpoint.pt','model.safetensors'];assert 'golden_sft' not in str(t) and 'swe-trace' not in str(t)
  st=t.stat();assert st.st_size==e['bytes'] and sha(t)==e['sha256'];assert not users(t)
  audit['files'].append(dict(e,target=str(t),source_is_symlink=p.is_symlink(),inode=st.st_ino,mtime_ns=st.st_mtime_ns,dev=st.st_dev,users=[]))
  seen.add(str(t));print('remote_verified',str(t),flush=True)
 audit['complete']=True;audit['finished']=time.time();save('pre_cleanup.json',audit);print('audit_complete',len(audit['files']),sum(e['bytes'] for e in audit['files']),flush=True)
else:
 assert sys.argv[1]=='clean' and len(sys.argv[2])>=7
 a=json.load((R/'pre_cleanup.json').open());assert a['complete'];before=protected()
 for e in a['files']:
  p=pathlib.Path(e['source']);t=pathlib.Path(e['target']);s=t.stat()
  assert p.resolve(strict=True)==t and s.st_size==e['bytes'] and s.st_ino==e['inode'] and s.st_mtime_ns==e['mtime_ns'] and not users(t)
 # Archive must still exist unchanged before removing any archived copies.
 ar=pathlib.Path('/data/patr-storage/patr-archives/patr-user-stop-20261014.tar.gz');assert ar.stat().st_size==32526413875
 r={'started':time.time(),'github_pre_sync':sys.argv[2],'before':disks(),'protected_before':before,'files':[]};save('receipt.json',r)
 for e in a['files']:
  p=pathlib.Path(e['source']);t=pathlib.Path(e['target']);assert not users(t)
  pointer=t.parent/(t.name+'.ARCHIVED.json');assert not pointer.exists()
  pointer.write_text(json.dumps(dict(e,restore_note='Exact bytes retained at local_backup, or extract archive_member from archive. Verify sha256 before use. Original run evidence remains in place.',github_pre_sync=sys.argv[2]),indent=2)+'\n')
  t.unlink()
  if p.is_symlink():p.unlink()
  assert not t.exists();r['files'].append(dict(e,restore_pointer=str(pointer),deleted_unix=time.time()));save('receipt.json',r)
 r.update(status='complete_verified',finished=time.time(),after=disks(),protected_after=protected());save('receipt.json',r)
 print(json.dumps({'status':r['status'],'bytes_freed':sum(e['bytes'] for e in r['files']),'after':r['after']}),flush=True)
