import json,os,time,subprocess,hashlib,shutil
from pathlib import Path
BASE=Path('/home/vipuser/storage-audit-20261104');OUT=BASE/'publish';REPO=BASE/'git-publish'
read=lambda p:json.loads(p.read_text())
pub=read(BASE/'publication_receipt.json');assert pub['remote_head_verified']
raw=(OUT/'deletion_manifest.json').read_bytes()
saved=subprocess.check_output(['git','-C',str(REPO),'show',pub['commit']+':'+pub['path']+'/deletion_manifest.json'])
assert saved==raw
plan=json.loads(raw);checks={r['run']:r for r in read(OUT/'verified_evaluations.json')}
def disk():return {p:dict(zip(('total','used','free'),shutil.disk_usage(p))) for p in ['/','/data']}
def open_targets():
 targets=set()
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=os.getuid():continue
   for fd in (proc/'fd').iterdir():
    try:targets.add(os.readlink(fd))
    except (OSError,PermissionError):pass
   try:
    for line in (proc/'maps').read_text().splitlines():
     cols=line.split(None,5)
     if len(cols)==6 and cols[-1].startswith('/'):targets.add(cols[-1])
   except (OSError,PermissionError):pass
  except (OSError,PermissionError):pass
 return targets
# Check entire fixed allowlist before removing any bytes.
used=open_targets()
for x in plan['files']:
 p=Path(x['path']);assert not p.is_symlink();st=p.stat()
 assert (st.st_size,st.st_ino,st.st_mtime_ns)==(x['bytes'],x['inode'],x['mtime_ns']),str(p)
 assert str(p) not in used,str(p)+' is open'
 run=Path(x['run']);s=read(run/'status.json');assert s['status'] in ('complete','smoke_complete')
 assert hashlib.sha256((run/'final_test.jsonl').read_bytes()).hexdigest()==checks[str(run)]['raw_sha256']
 assert '/runs/formal/tree_random_seed43/' not in str(p)
 assert p.name=='checkpoint.pt' or (p.parent.name=='final_model' and ('/smoke/' in str(p) or str(run) in ['/data/patr-storage/v1-tree-20261016/runs/formal/patr_seed42','/home/vipuser/patr-v1-tree-results/tree_random_seed42']))
receipt=dict(started=time.time(),published_manifest_commit=pub['commit'],before=disk(),deleted=[],skipped=[])
def save():
 tmp=OUT/'cleanup_receipt.tmp';tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(OUT/'cleanup_receipt.json')
save()
for x in plan['files']:
 p=Path(x['path']);st=p.stat();assert (st.st_size,st.st_ino,st.st_mtime_ns)==(x['bytes'],x['inode'],x['mtime_ns'])
 assert str(p) not in open_targets()
 assert read(Path(x['run'])/'status.json')['status'] in ('complete','smoke_complete')
 p.unlink()
 receipt['deleted'].append(dict(path=x['path'],bytes=x['bytes'],sha256=x['sha256'],deleted_at=time.time(),disk_after=disk()));save()
receipt.update(completed=time.time(),deleted_bytes=sum(x['bytes'] for x in receipt['deleted']),after=disk());save()
print(json.dumps(dict(files=len(receipt['deleted']),deleted_bytes=receipt['deleted_bytes'],before=receipt['before'],after=receipt['after'])))
