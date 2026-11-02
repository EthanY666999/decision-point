"""Verify complete local backups and archive lightweight evidence before cleanup."""
import hashlib, json, subprocess, time
from pathlib import Path

BASE = Path('/Users/ethan/Documents/ChatGPT/ACL2027')
OUT = BASE / 'output/golden-cert-20261102-r1/disk_20261103'
OUT.mkdir(parents=True, exist_ok=True)
OLD = BASE / 'output/golden-next-20261031'
NEW = BASE / 'output/golden-cert-20261102-r1'
HOME = '/home/vipuser/'
SPECS = [
    ('sft_grpo400', OLD/'weights-backup', HOME+'golden-sft-patr-20261031/code/runs/formal/grpo_seed42'),
    ('sft_patr400', OLD/'weights-backup', HOME+'golden-sft-patr-20261031/code/runs/formal/patr_seed42'),
    ('sft_tree_random400', OLD/'weights-backup', HOME+'golden-controls-20261031/code/runs/formal/tree_random_seed42'),
    ('sft_grpo4_400', OLD/'weights-backup-grpo4', HOME+'golden-controls-20261031/code/runs/formal/grpo_seed42'),
    ('b3_patr_formal', OLD/'weights-backup-b3_patr_formal', HOME+'golden-next-20261031/b3/code/runs/formal/patr_seed42'),
    ('cert_r1_uct_400', NEW/'weights-backup-cert_r1_uct_400', HOME+'golden-cert-20261102-r1/code/runs/g2/uct/code/runs/formal/patr_seed42'),
    ('cert_r1_uniform_400', NEW/'weights-backup-cert_r1_uniform_400', HOME+'golden-cert-20261102-r1/code/runs/g2/uniform/code/runs/formal/patr_seed42'),
]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''):h.update(b)
    return h.hexdigest()

result={'created_unix':time.time(), 'files':[], 'scope':'Only seven completed redundant checkpoint replicas; final models and all evidence retained.'}
for label, backup, run in SPECS:
    status=json.loads((backup/'status.json').read_text())
    assert status['status']=='complete_verified', (label,status)
    receipt=json.loads((backup/'receipt.json').read_text())
    entries={k:v for k,v in receipt.items() if k.startswith(label+'/')}
    assert label+'/checkpoint.pt' in entries and label+'/final_model/model.safetensors' in entries
    checked=[]
    for name,v in entries.items():
        p=backup/name
        assert v['verified'] and p.stat().st_size==v['bytes'] and digest(p)==v['sha256'], str(p)
        checked.append(name)
    cp=entries[label+'/checkpoint.pt']
    result['files'].append({'label':label, 'run':run, 'source':run+'/checkpoint.pt', 'local_backup':str(backup/label/'checkpoint.pt'), 'bytes':cp['bytes'], 'sha256':cp['sha256'], 'all_local_files_verified':checked, 'local_verified_unix':time.time(), 'expected_steps':400})
    (OUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('local_verified',label,len(checked),flush=True)
    dest=OUT/'light_runs'/label;dest.mkdir(parents=True,exist_ok=True)
    for attempt in range(5):
        copied=subprocess.run(['rsync','-a','-e','ssh -o IPQoS=none -o ConnectTimeout=15','--exclude=checkpoint*','--exclude=final_model/','--exclude=*.pt','--exclude=*.safetensors','galaxy-4090:'+run+'/',str(dest)+'/'])
        if copied.returncode==0:break
        time.sleep(2)
    else:raise RuntimeError('Light archive failed: '+label)
result['complete']=True
(OUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print('all_verified',len(result['files']),sum(x['bytes'] for x in result['files']),flush=True)
