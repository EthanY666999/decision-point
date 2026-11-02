import tarfile,hashlib,json,pathlib,time,sys
out=pathlib.Path('/home/vipuser/golden-cert-20261102-r1/code/artifacts/deep_disk_20261103');out.mkdir(exist_ok=True)
def digest(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
if sys.argv[1]=='archive':
 p='/data/patr-storage/patr-archives/patr-user-stop-20261014.tar.gz'
 r={'archive':p,'started':time.time(),'large_members':[],'members':0}
 with tarfile.open(p,'r|gz') as t:
  for m in t:
   r['members']+=1
   if m.isfile() and m.size>1000000000:
    sha=digest(t.extractfile(m));r['large_members'].append({'name':m.name,'bytes':m.size,'sha256':sha});print(m.name,m.size,sha,flush=True)
    (out/'archive_members_progress.json').write_text(json.dumps(r,indent=2))
 r['complete']=True;r['finished']=time.time();(out/'archive_members.json').write_text(json.dumps(r,indent=2));print('complete',r['members'],flush=True)
else:
 roots=['/home/vipuser/patr-reproduction/runs','/home/vipuser/patr-v1-tree-results','/home/vipuser/golden-sft-patr-20261031/code/runs/smoke','/home/vipuser/golden-controls-20261031/code/runs/smoke','/data/patr-storage/v1-tree-20261016/runs']
 r=[]
 for root in roots:
  for p in pathlib.Path(root).rglob('*'):
   if p.is_symlink() or not p.is_file() or p.stat().st_size<1000000000:continue
   s=p.stat()
   with p.open('rb') as f:sha=digest(f)
   r.append({'path':str(p),'bytes':s.st_size,'sha256':sha,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'dev':s.st_dev});print(str(p),sha,flush=True)
   (out/'large_files.json').write_text(json.dumps(r,indent=2))
 print('complete',len(r),flush=True)
