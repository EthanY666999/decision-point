"""Download pinned TRAIN data and build an auditable, decontaminated manifest."""
import hashlib
import json
import pathlib
import random
import re
from collections import Counter
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
REV = '87fb43eb4d6c71b1e86292346e45f52b71ea7691'
SOURCE = 'zhenghaoxu/R2E-Gym-Lite-with-Difficulty'

def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()

def textkey(s):
    return digest(re.sub(r'\s+', ' ', s).strip().casefold())

def download(repo, rev, name):
    return hf_hub_download(repo, name, repo_type='dataset', revision=rev,
                          local_dir=DATA / repo.split('/')[-1])

def main():
    held = download('princeton-nlp/SWE-bench_Verified',
                    'c104f840cc67f8b6eec6f759ebc8b2693d585d4a',
                    'data/test-00000-of-00001.parquet')
    heldrows = pq.read_table(held).to_pylist()
    blocked_ids = {r['instance_id'] for r in heldrows}
    blocked_text = {textkey(r['problem_statement']) for r in heldrows}
    blocked_commit = {r['base_commit'] for r in heldrows}
    excluded = Counter()
    seen = set()
    candidates = []
    for shard in range(3):
        name = f'data/filtered_1_7-{shard:05d}-of-00003.parquet'
        path = download(SOURCE, REV, name)
        table = pq.read_table(path, columns=['instance_id','repo_name','commit_hash',
                                            'docker_image','problem_statement','difficulty'])
        for idx, row in enumerate(table.to_pylist()):
            key = (row['repo_name'], row['commit_hash'])
            reason = None
            if row['instance_id'] in blocked_ids or textkey(row['problem_statement']) in blocked_text or row['commit_hash'] in blocked_commit:
                reason = 'heldout_exact_id_text_or_commit'
            elif key in seen:
                reason = 'duplicate_repo_commit'
            if reason:
                excluded[reason] += 1
                continue
            seen.add(key)
            candidates.append({**row, 'shard':name, 'row_index':idx,
                               'problem_sha256':digest(row['problem_statement'])})
        print(json.dumps({'shard':shard,'eligible_so_far':len(candidates)}), flush=True)
    random.Random(42).shuffle(candidates)
    (ROOT/'candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2)+'\n')
    report = {'source':SOURCE,'revision':REV,'split':'filtered_1_7',
              'eligible':len(candidates),'excluded':dict(excluded),'selection_seed':42,
              'heldout_rows':len(heldrows),
              'overlap_check':'exact instance ID, whitespace-normalized issue, and commit hash; not a proof of semantic non-overlap',
              'target':1000,'selection':'fixed shuffle; retain independently verified successes; no duplicate instances'}
    (ROOT/'data_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__ == '__main__':
    main()
