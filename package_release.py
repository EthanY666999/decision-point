"""Package completed remote data, without weights or local backups."""
import collections
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path('/home/vipuser/golden-flh-trace-20261108')
OUT = ROOT / 'release'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def main():
    status = json.loads((ROOT / 'status.json').read_text())
    assert status['state'] == 'complete' and status['independent_final_replay']
    summary = json.loads((OUT / 'summary.json').read_text())
    assert summary['independently_replayed'] == 1000
    data = json.loads((ROOT / 'data.json').read_text())
    holdout = data['validation'] + data['test'] + data['supplemental']
    write('heldout_map_hashes.json', {'uid': [t['uid'] for t in holdout], 'map_sha256': [t['map_sha256'] for t in holdout]})
    train = data['train']
    write('train_map_hashes.json', {t['uid']: t['map_sha256'] for t in train})
    for name in ['manifest.json', 'pilot.json', 'remote_dataset.py', 'teacher_bridge.py', 'response_schema.json', 'verify_release.py', 'package_release.py', 'audit_token_lengths.py', 'README.md']:
        shutil.copyfile(ROOT / name, OUT / name)
    shutil.copytree(ROOT / 'source', OUT / 'source', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    (OUT / 'requirements.txt').write_text('numpy==2.2.6\ngymnasium==1.1.1\n')
    failures = collections.Counter()
    with gzip.open(OUT / 'failed_episodes.jsonl.gz', 'wt', encoding='utf-8') as f:
        for p in sorted((ROOT / 'episodes').glob('*.json')):
            row = json.loads(p.read_text())
            if not row['accepted']:
                failures[row['terminal']] += 1
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    raw = []
    usage = collections.Counter()
    teacher_seconds = 0.0
    with gzip.open(OUT / 'teacher_calls.jsonl.gz', 'wt', encoding='utf-8') as f:
        for p in sorted((ROOT / 'provenance').glob('b*.json')):
            packet = json.loads(p.read_text())
            events = [json.loads(line) for line in packet['stdout'].splitlines() if line.strip()]
            answers = [e['item']['text'] for e in events if e.get('type') == 'item.completed' and e.get('item', {}).get('type') == 'agent_message']
            uses = [e.get('usage', {}) for e in events if e.get('type') == 'turn.completed']
            for u in uses:
                usage.update({k: v for k, v in u.items() if isinstance(v, (int, float))})
            teacher_seconds += packet['seconds']
            raw.append(dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p)))
            # Publish actual instructions, maps, final output and token usage.
            # Keep CLI stderr and local machine paths in original remote evidence only.
            f.write(json.dumps(dict(request=packet['request'], requested_model=packet['requested_model'],
                                   fallback_model=None, call_id=packet['call_id'], response_text=answers[-1] if answers else None,
                                   returncode=packet['returncode'], seconds=packet['seconds'], usage=uses,
                                   raw_packet_sha256=digest(p)), ensure_ascii=False) + '\n')
    write('remote_evidence_inventory.json', {'raw_teacher_calls': raw, 'model_probe_counted_as_training': False})
    summary.update(failure_categories=dict(failures), teacher_call_count=len(raw), teacher_seconds=teacher_seconds, teacher_token_usage=dict(usage))
    summary['files'] = {str(p.relative_to(OUT)): dict(bytes=p.stat().st_size, sha256=digest(p))
                        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['summary.json', 'independent_validation.json'] and '__pycache__' not in p.parts}
    write('summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == '__main__':
    main()
