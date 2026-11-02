"""Sync r1 evidence without weights; losslessly gzip growing branch logs for Git."""
from pathlib import Path
import subprocess,gzip,hashlib,json,shutil,sys
base=Path(__file__).resolve().parents[2];work=base/'work/golden-cert-20261102-r1';out=base/'output/golden-cert-20261102-r1';repo=base/'work/golden-sft-patr-run-20261031';dest=repo/'experiments/golden-cert-20261102-r1'
subprocess.run(['rsync','-a','--exclude=__pycache__/','--exclude=*.pyc','--exclude=vendor',str(work)+'/',str(dest)+'/'],check=True)
subprocess.run(['rsync','-a','--exclude=weights-backup*','--exclude=checkpoint*','--exclude=final_model/','--exclude=*.pt','--exclude=*.safetensors','--exclude=branch_events.jsonl',str(out)+'/',str(dest/'evidence')+'/'],check=True)
receipts=[]
for src in out.rglob('branch_events.jsonl'):
 target=dest/'evidence'/src.relative_to(out);packed=target.with_suffix(target.suffix+'.gz');packed.parent.mkdir(parents=True,exist_ok=True)
 raw=src.read_bytes();compressed=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(compressed)==raw;packed.write_bytes(compressed)
 # Remove only the redundant Git working-copy text; original local and remote are retained.
 if target.exists():target.unlink()
 receipts.append(dict(source=str(src.relative_to(base)),compressed=str(packed.relative_to(repo)),raw_bytes=len(raw),compressed_bytes=len(compressed),raw_sha256=hashlib.sha256(raw).hexdigest(),roundtrip_verified=True))
(dest/'evidence/branch_log_compression.json').write_text(json.dumps(receipts,indent=2)+'\n')
p=repo/'GOLDEN_SYNC_MANIFEST.json';m=json.loads(p.read_text());m['snapshot']=sys.argv[1] if len(sys.argv)>1 else 'CERT r1 light evidence with lossless compressed branch logs'
for n in list(m['files']):
 if n.startswith('experiments/golden-cert-20261102-r1/') and n.endswith('/branch_events.jsonl'):del m['files'][n]
for f in dest.rglob('*'):
 if f.is_file() and not f.is_symlink():m['files'][str(f.relative_to(repo))]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(receipts))
