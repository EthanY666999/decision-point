import json,hashlib,datetime
from pathlib import Path
root=Path(__file__).resolve().parent
published=root/'publication/patr-v1-tree-closeout-20261017'
integrity=json.loads((published/'analysis/integrity.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
result=dict(verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Two final inference models and tokenizer/config metadata copied locally. Optimizer checkpoints and raw training rollouts remain remote; no weights uploaded to GitHub.',models={})
for method in ['patr','tree_random']:
 folder=root/'model-backup'/f'{method}_seed42'
 h=sha(folder/'model.safetensors');assert h==integrity['models'][method]['final_model_sha256']
 for p in (published/'runs'/f'{method}_seed42'/'final_model_metadata').iterdir():assert sha(p)==sha(folder/p.name),p.name
 result['models'][method]=dict(relative_directory=str(folder.relative_to(root)),sha256=h,files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.iterdir()) if p.is_file()])
(root/'LOCAL_MODEL_BACKUP_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({m:d['sha256'] for m,d in result['models'].items()}))
