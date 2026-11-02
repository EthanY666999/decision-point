import pathlib,json,hashlib,time,subprocess
B=pathlib.Path('/Users/ethan/Documents/ChatGPT/ACL2027');O=B/'output/golden-cert-20261102-r1/deep_disk_20261103';O.mkdir(exist_ok=True)
H='/home/vipuser/'; roots={'sft_grpo400':H+'golden-sft-patr-20261031/code/runs/formal/grpo_seed42','sft_patr400':H+'golden-sft-patr-20261031/code/runs/formal/patr_seed42','sft_patr500':H+'golden-patr-extended800-20261031/code/runs/formal/patr_seed42','sft_tree_random400':H+'golden-controls-20261031/code/runs/formal/tree_random_seed42','sft_grpo4_400':H+'golden-controls-20261031/code/runs/formal/grpo_seed42'}
for group in ['b2','b3']:
 for method in ['patr','tree_random']:
  for stage in ['smoke','formal']: roots[f'{group}_{method}_{stage}']=H+f'golden-next-20261031/{group}/code/runs/{stage}/{method}_seed42'
for method in ['patr','tree_random']:
 for stage in ['smoke','formal']:roots[f'seed43_{method}_{stage}']=H+f'golden-next-20261031/seed43_{method}/code/runs/{stage}/{method}_seed43'
for s in ['uct','uniform']:roots[f'cert_r1_{s}_400']=H+f'golden-cert-20261102-r1/code/runs/g2/{s}/code/runs/formal/patr_seed42'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
r={'created':time.time(),'scope':'Remove only redundant server weight copies with freshly verified complete local backup; keep all evidence, SFT and current projects.','files':[],'packages':[]}
for parent in ['golden-next-20261031','golden-cert-20261102-r1']:
 for rp in (B/'output'/parent).glob('weights-backup*/receipt.json'):
  d=json.load(rp.open()); assert json.load((rp.parent/'status.json').open())['status']=='complete_verified'
  for label in sorted(set(k.split('/')[0] for k in d)-{'sft'}):
   if label not in roots:continue
   entries={k:v for k,v in d.items() if k.startswith(label+'/')};assert label+'/checkpoint.pt' in entries
   for name,v in entries.items():
    p=rp.parent/name;assert v['verified'] and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],str(p)
   r['packages'].append({'label':label,'local_root':str(rp.parent/label),'run':roots[label],'all_files':entries,'verified_unix':time.time()})
   for rel in ['final_model/model.safetensors']+(['checkpoint.pt'] if label=='sft_patr500' else []):
    v=entries[label+'/'+rel];r['files'].append(dict(label=label,source=roots[label]+'/'+rel,local_backup=str(rp.parent/label/rel),bytes=v['bytes'],sha256=v['sha256'],backup_kind='local_complete_package'))
   (O/'local_candidates.json').write_text(json.dumps(r,indent=2));print('full_local_verified',label,len(entries),flush=True)
r['complete']=True;(O/'local_candidates.json').write_text(json.dumps(r,indent=2));print('done',len(r['packages']),flush=True)
