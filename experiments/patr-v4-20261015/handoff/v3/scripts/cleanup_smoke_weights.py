"""Remove only disposable engineering/smoke weights, never formal checkpoints."""
import argparse,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
roots=[ROOT/'runs/smoke',ROOT/'runs/engineering/smoke_batched']
candidates=[]
for root in roots:
    if not root.exists():continue
    for run in root.iterdir():
        status=run/'status.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='smoke_complete':
            candidates.extend([run/'checkpoint.pt',run/'checkpoint.tmp',run/'final_model'])
calibration=ROOT/'runs/engineering/calibration_batched'
if (calibration/'status.json').exists():
    s=json.loads((calibration/'status.json').read_text())
    if s.get('status')=='paused_checkpointed' and s.get('step')==0:candidates.append(calibration/'checkpoint.pt')
items=[]
for path in candidates:
    if not path.exists():continue
    assert 'formal' not in path.relative_to(ROOT).parts
    size=sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.is_dir() else path.stat().st_size
    items.append(dict(path=str(path.relative_to(ROOT)),bytes=size))
    if args.apply:
        if path.is_dir():shutil.rmtree(path)
        else:path.unlink()
record=dict(unix=time.time(),applied=args.apply,items=items,reason='Disposable smoke weights removed; metrics, logs, raw trajectories, sources preserved; formal checkpoints untouched')
print(json.dumps(record))
if args.apply:
    with (ROOT/'artifacts/smoke_weight_cleanup.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
