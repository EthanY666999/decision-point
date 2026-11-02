"""Run on the Mac: pull immutable completed model/checkpoint files, verify SHA256.

No GPU use, remote mutation, deletion, or copying of a live training checkpoint.
Serial, resumable, bandwidth limited. A partial backup is never marked verified.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

OUT=Path('/Users/ethan/Documents/ChatGPT/ACL2027/output/golden-next-20261031/weights-backup')
ROOTS={
 'sft':'/home/vipuser/golden-sft-patr-20261031/code/runs/golden_sft/sft',
 'sft_grpo400':'/home/vipuser/golden-sft-patr-20261031/code/runs/formal/grpo_seed42',
 'sft_patr400':'/home/vipuser/golden-sft-patr-20261031/code/runs/formal/patr_seed42',
 'sft_patr500':'/home/vipuser/golden-patr-extended800-20261031/code/runs/formal/patr_seed42',
 'sft_tree_random400':'/home/vipuser/golden-controls-20261031/code/runs/formal/tree_random_seed42',
}

def save(path,obj):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(path)

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'backup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt=json.loads((OUT/'receipt.json').read_text()) if (OUT/'receipt.json').exists() else {}
 for label,root in ROOTS.items():
  save(OUT/'status.json',dict(status='hashing_remote',label=label,time=time.time()))
  # Hash immutable files serially. The script is passed over stdin, not shell interpolation.
  code="""import json,hashlib
from pathlib import Path
r=Path(ROOT)
files=sorted(p for p in (r/'final_model').rglob('*') if p.is_file())
if (r/'checkpoint.pt').exists():files.append(r/'checkpoint.pt')
out={}
for p in files:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 out[str(p.relative_to(r))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps(out))
""".replace('Path(ROOT)', 'Path('+repr(root)+')')
  result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','galaxy-4090','python3 -'],input=code,text=True,capture_output=True,check=True)
  manifest=json.loads(result.stdout); dest=OUT/label;dest.mkdir(exist_ok=True)
  save(dest/'remote_manifest.json',manifest)
  for name,expected in manifest.items():
   local=dest/name;local.parent.mkdir(parents=True,exist_ok=True)
   key=label+'/'+name
   if local.exists() and local.stat().st_size==expected['bytes'] and digest(local)==expected['sha256']:
    receipt[key]=dict(**expected,verified=True);save(OUT/'receipt.json',receipt);continue
   save(OUT/'status.json',dict(status='transferring',label=label,file=name,bytes=expected['bytes'],time=time.time()))
   subprocess.run(['rsync','-a','--partial','--bwlimit=12000','--timeout=120','-e','ssh -o BatchMode=yes -o ConnectTimeout=20',
                   'galaxy-4090:'+root+'/'+name,str(local)],check=True)
   assert local.stat().st_size==expected['bytes'] and digest(local)==expected['sha256'], 'Backup integrity mismatch: '+key
   receipt[key]=dict(**expected,verified=True,verified_unix=time.time());save(OUT/'receipt.json',receipt)
 save(OUT/'status.json',dict(status='complete_verified',files=len(receipt),bytes=sum(x['bytes'] for x in receipt.values()),time=time.time()))

if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():save(OUT/'status.json',dict(status='needs_recovery',error=repr(e),time=time.time()))
  raise
