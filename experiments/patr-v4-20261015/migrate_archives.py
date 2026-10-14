from pathlib import Path
import os,shutil,hashlib,json,time,signal,tarfile
src=Path('/home/vipuser/patr-archives');dst=Path('/data/patr-storage/patr-archives');receipt=src/'migration_v4_manifest.json'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
assert os.path.ismount('/data') and os.stat('/data').st_dev!=os.stat('/home/vipuser').st_dev
pid=697037
if Path(f'/proc/{pid}/cmdline').exists():
 assert b'/home/vipuser/patr-archives/upload_v1.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
 os.killpg(pid,signal.SIGKILL)
items=[src/'patr-user-stop-20261014.tar.gz']+sorted((src/'release-parts-20261014').iterdir())
assert all(x.is_file() and not x.is_symlink() for x in items)
size=sum(p.stat().st_size for p in items)
assert shutil.disk_usage('/data').free-size>25*1024**3
rows=[]
for p in items:
 target=dst/p.relative_to(src);target.parent.mkdir(parents=True,exist_ok=True)
 before=p.stat();shutil.copy2(p,target)
 a=digest(p);b=digest(target);after=p.stat()
 assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and a==b
 rows.append({'source':str(p),'target':str(target),'bytes':before.st_size,'sha256':a,'verified_at':time.time(),'source_removed':False})
 print('VERIFIED',p.name,before.st_size,flush=True)
 receipt.write_text(json.dumps({'state':'copying_verified','files':rows},indent=2))
expected=json.loads((dst/'release-parts-20261014/ARCHIVE.json').read_text())
assert rows[0]['sha256']==expected['sha256']
for part in expected['parts']:
 r=next(r for r in rows if Path(r['source']).name==part['file']);assert r['sha256']==part['sha256'] and r['bytes']==part['bytes']
n=0
with tarfile.open(dst/'patr-user-stop-20261014.tar.gz','r|gz') as t:
 for m in t:
  if m.isfile():
   f=t.extractfile(m)
   while f.read(8*1024*1024):pass
  n+=1
print('ARCHIVE_READABLE',n,flush=True)
for r in rows:
 p=Path(r['source']);assert p.stat().st_size==r['bytes'];p.unlink();r['source_removed']=True
 receipt.write_text(json.dumps({'state':'removing_verified_sources','archive_members':n,'files':rows},indent=2))
(src/'release-parts-20261014').rmdir()
(src/'release-parts-20261014').symlink_to(dst/'release-parts-20261014',target_is_directory=True)
(src/'patr-user-stop-20261014.tar.gz').symlink_to(dst/'patr-user-stop-20261014.tar.gz')
r={'state':'complete_verified','archive_members':n,'source_bytes_released':size,'files':rows,'finished':time.time(),'data_free':shutil.disk_usage('/data').free,'system_free':shutil.disk_usage('/').free}
receipt.write_text(json.dumps(r,indent=2));(dst/'migration_v4_manifest.json').write_text(json.dumps(r,indent=2))
print('COMPLETE',size,flush=True)
