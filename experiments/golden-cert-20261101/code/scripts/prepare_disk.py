import hashlib,json,time
from pathlib import Path
base=Path('/Users/ethan/Documents/ChatGPT/ACL2027')
out=base/'output/golden-cert-20261101/disk_20261101'
entries=[]
for label in ['b2_patr_formal','b2_tree_random_formal','b2_tree_random_smoke','b3_patr_smoke']:
 b=base/'output/golden-next-20261031'/('weights-backup-'+label)
 assert json.loads((b/'status.json').read_text())['status']=='complete_verified'
 stage=label.rsplit('_',1)[1];generation=label.split('_')[0];method=label[len(generation)+1:-(len(stage)+1)]
 root='/home/vipuser/golden-next-20261031/'+generation+'/code/runs/'+stage+'/'+method+'_seed42'
 manifest=json.loads((b/label/'remote_manifest.json').read_text())
 for name,v in manifest.items():
  p=b/label/name
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
  assert p.stat().st_size==v['bytes'] and h.hexdigest()==v['sha256']
  action='archive_remote_replica' if stage=='smoke' or name=='checkpoint.pt' else 'migrate'
  entries.append(dict(source=root+'/'+name,local_backup=str(p),run=root,expected_status='smoke_complete' if stage=='smoke' else 'complete',expected_steps=1 if stage=='smoke' else 150,action=action,destination='/data/patr-storage/cert-space-20261101/'+generation+'/'+method+'/'+name if action=='migrate' else None,**v))
  print(label,name,'verified',flush=True)
(out/'manifest.json').write_text(json.dumps(dict(time=time.time(),files=entries,archive_policy='Only redundant remote replicas; complete local SHA verified bytes retained; no logs, trajectories, configs or formal final models deleted'),indent=2)+'\n')
print('READY',flush=True)
