"""Download the frozen public model revision, with LFS SHA256 verification."""
import hashlib,json,subprocess,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
# Deliberately uses the standard library so it works before ML packages are installed.
ROOT=Path(__file__).resolve().parents[1]
REPO='Qwen/Qwen2.5-0.5B-Instruct'
REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
WANTED={'config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','LICENSE','README.md'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true',timeout=30) as r:meta=json.load(r)
    assert meta['sha']==REVISION
    dest=ROOT/'artifacts/Qwen2.5-0.5B-Instruct';dest.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for entry in meta['siblings']:
        name=entry['rfilename']
        if name not in WANTED:continue
        path=dest/name
        expected=entry.get('lfs',{}).get('sha256')
        if not path.exists() or expected and sha(path)!=expected:
            temp=path.with_suffix(path.suffix+'.part')
            subprocess.run(['curl','-fL','--retry','4','--connect-timeout','20','--max-time','1800','--output',str(temp),f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'],check=True)
            if expected:assert sha(temp)==expected, f'Hash mismatch: {name}'
            if entry.get('size'):assert temp.stat().st_size==entry['size']
            temp.replace(path)
        manifest.append(dict(path=name,sha256=sha(path),bytes=path.stat().st_size))
    assert {r['path'] for r in manifest}==WANTED
    (ROOT/'artifacts/model_manifest.json').write_text(json.dumps(dict(repo=REPO,revision=REVISION,files=manifest),indent=2)+'\n')
    print('MODEL_DOWNLOAD_COMPLETE')
if __name__=='__main__':main()
