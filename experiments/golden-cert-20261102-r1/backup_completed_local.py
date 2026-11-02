"""Back up a completed B3 run on the Mac; never delete remote artifacts."""
import argparse, fcntl, hashlib, json, subprocess, time
from pathlib import Path
import sys
sys.path.insert(0,'/Users/ethan/Documents/ChatGPT/ACL2027/work/golden-next-20261031/code/scripts')
import backup_completed_local as engine


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selector', choices=['uct','uniform'], required=True)
    
    a=p.parse_args()
    label=f'cert_r1_{a.selector}_400'
    root=f'/home/vipuser/golden-cert-20261102-r1/code/runs/g2/{a.selector}/code/runs/formal/patr_seed42'
    out=Path('/Users/ethan/Documents/ChatGPT/ACL2027/output/golden-cert-20261102-r1')/f'weights-backup-{label}'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'backup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        engine.save(out/'status.json',dict(status='hashing_remote',time=time.time()))
        code='''import json,hashlib
from pathlib import Path
r=Path(ROOT);s=json.loads((r/'status.json').read_text())
assert s['status']==STATE and s['step']==s['total_steps']==COUNT,s
assert (r/'final_test.json').exists()
files=[r/'checkpoint.pt']+sorted((r/'final_model').rglob('*'))
x={}
for f in files:
 if not f.is_file():continue
 assert not f.is_symlink()
 h=hashlib.sha256()
 with f.open('rb') as src:
  for b in iter(lambda:src.read(8*1024*1024),b''):h.update(b)
 x[str(f.relative_to(r))]={'bytes':f.stat().st_size,'sha256':h.hexdigest()}
assert 'checkpoint.pt' in x and 'final_model/model.safetensors' in x
print(json.dumps(x))
'''.replace('ROOT',repr(root)).replace('STATE',repr('complete')).replace('COUNT',str(400))
        manifest=json.loads(subprocess.run(['ssh','galaxy-4090','python3 -'],input=code,text=True,capture_output=True,check=True).stdout)
        dest=out/label;dest.mkdir(exist_ok=True);engine.save(dest/'remote_manifest.json',manifest)
        receipt=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
        for name,v in manifest.items():
            local=dest/name;local.parent.mkdir(parents=True,exist_ok=True)
            if not(local.exists() and local.stat().st_size==v['bytes'] and engine.digest(local)==v['sha256']):
                engine.save(out/'status.json',dict(status='transferring',file=name,time=time.time(),bandwidth_KiBps=60000))
                subprocess.run(['rsync','-a','--partial','--bwlimit=60000','--timeout=120','-e','ssh -o BatchMode=yes -o ConnectTimeout=20','galaxy-4090:'+root+'/'+name,str(local)],check=True)
            assert local.stat().st_size==v['bytes'] and engine.digest(local)==v['sha256']
            receipt[label+'/'+name]=dict(**v,verified=True,verified_unix=time.time());engine.save(out/'receipt.json',receipt)
        engine.save(out/'status.json',dict(status='complete_verified',files=len(receipt),bytes=sum(v['bytes'] for v in receipt.values()),time=time.time()))
    except Exception as e:
        engine.save(out/'status.json',dict(status='needs_recovery',error=repr(e),time=time.time()));raise

if __name__=='__main__':main()
