import json,tarfile,hashlib,gzip,shutil,time,subprocess,importlib.metadata
from pathlib import Path
root=Path('/home/vipuser/patr-v1-tree-20261016');out=root/'reports/closeout-20261017';base=Path('/home/vipuser/patr-reproduction')
assert json.loads((out/'evaluation_status.json').read_text())['phase']=='complete'
stage=out/'archive-published';stage.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def copy(src,dst):
 dst=stage/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for p in root.glob('*.py'):copy(p,Path('source')/p.name)
for n in ['configs/frozenlake_05b.json','artifacts/data.json','artifacts/runtime_override.json','artifacts/deadline.json']:
 copy(root/n,Path('source')/n)
for p in (root/'vendor/rllm_source').rglob('*.py'):copy(p,Path('source/vendor/rllm_source')/p.relative_to(root/'vendor/rllm_source'))
for p in out.iterdir():
 if p.is_file() and p.suffix in ['.json','.jsonl','.py','.log','.txt','.png','.svg']:copy(p,Path('analysis')/p.name)
for p in (root/'reports').iterdir():
 if p.is_file() and p.suffix in ['.json','.jsonl','.log']:copy(p,Path('reports')/p.name)
for dirname in ['grpo_diagnostic','stage1']:
 folder=root/'reports'/dirname
 if folder.exists():
  for p in folder.rglob('*'):
   if p.is_file() and p.suffix in ['.json','.jsonl','.gz','.txt']:copy(p,Path('reports')/dirname/p.relative_to(folder))
manifest=[]
for method in ['patr','tree_random']:
 run=root/'runs/formal'/f'{method}_seed42';dst=Path('runs')/f'{method}_seed42'
 for p in run.iterdir():
  if p.is_file() and p.suffix in ['.json','.jsonl']:
   if p.name=='branch_events.jsonl':
    (stage/dst).mkdir(exist_ok=True,parents=True)
    with p.open('rb') as a,gzip.open(stage/dst/(p.name+'.gz'),'wb') as b:shutil.copyfileobj(a,b)
   else:copy(p,dst/p.name)
 for p in (run/'source_snapshots').rglob('*.py'):copy(p,dst/'source_snapshots'/p.relative_to(run/'source_snapshots'))
 for p in (run/'final_model').iterdir():
  if p.is_file() and p.suffix in ['.json','.txt']:copy(p,dst/'final_model_metadata'/p.name)
 for p in sorted(run.glob('rollouts_*.jsonl.gz')):
  manifest.append(dict(method=method,path=str(p),size=p.stat().st_size,sha256=sha(p),storage='Retained on remote; not included in lightweight Git archive'))
for method in ['grpo','patr','tree_random']:
 run=base/'runs/formal'/f'{method}_seed42'
 for n in ['final_test.json','final_test.jsonl','config.json','status.json','source_versions.jsonl']:
  if (run/n).exists():copy(run/n,Path('v1_baselines')/f'{method}_seed42'/n)
(stage/'raw_rollout_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
versions={}
for name in ['torch','vllm','xformers','transformers','flash-attn','numpy','gymnasium']:
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]=None
versions['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total','--format=csv,noheader'],text=True).strip()
versions['python']=subprocess.check_output(['/home/vipuser/patr-reproduction/.venv/bin/python','--version'],text=True).strip()
(stage/'runtime_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
files=[dict(path=str(p.relative_to(stage)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(stage.rglob('*')) if p.is_file()]
(stage/'MANIFEST.json').write_text(json.dumps(dict(created_unix=time.time(),files=files,exclusions='No model weights, optimizer states, secrets or cache. 800 raw training rollout files retained remotely with SHA256 manifest. Full tests, validation trajectories, metrics, tree events, source and diagnostic evidence included.'),indent=2)+'\n')
tar=out/'patr-v1-tree-closeout-20261017-published.tar.gz'
with tarfile.open(tar,'w:gz') as t:t.add(stage,arcname='patr-v1-tree-closeout-20261017')
receipt=dict(path=str(tar),bytes=tar.stat().st_size,sha256=sha(tar),file_count=len(files)+1)
(out/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
