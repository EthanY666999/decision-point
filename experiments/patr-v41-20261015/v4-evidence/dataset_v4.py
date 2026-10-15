"""Freeze legacy generation (8) independently of interaction horizon (10)."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/rllm_source'))
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv, generate_random_map, normalize_desc, shortest_path_length


def rows(seeds, sizes, probabilities, split):
    return [dict(seed=int(s), size=int(z), p=float(p), uid=f'{s}_{z}_{p}', split=split)
            for s, z, p in zip(seeds, sizes, probabilities)]


def parameters():
    rng = np.random.RandomState(42)
    ts, es = rng.randint(0, 100000, 10000), rng.randint(0, 100000, 100)
    tz, ez = rng.randint(2, 10, 10000), rng.randint(2, 10, 100)
    tp, ep = rng.uniform(.6, .85, 10000), rng.uniform(.6, .85, 100)
    data = dict(train=rows(ts, tz, tp, 'train'), test=rows(es, ez, ep, 'test'))
    vr = np.random.RandomState(20261013)
    data['validation'] = rows(vr.randint(100000, 200000, 100), vr.randint(2, 10, 100), vr.uniform(.6, .85, 100), 'validation')
    # Independent dataset RNG is not an additional training seed. Freeze before evaluation.
    hr = np.random.RandomState(20261015)
    data['holdout'] = rows(hr.randint(200000, 300000, 100), hr.randint(2, 10, 100), hr.uniform(.6, .85, 100), 'holdout')
    return data


def materialize(task):
    desc, _ = generate_random_map(size=task['size'], p=task['p'], seed=task['seed'], max_steps=8, generation_mode='legacy')
    desc = normalize_desc(desc)
    digest = hashlib.sha256('\n'.join(desc).encode()).hexdigest()
    return dict(task, desc=desc, map_sha256=digest, shortest_path=shortest_path_length(desc), generation_max_steps=8, generation_mode='legacy')


def main():
    destination = ROOT / 'artifacts/data_v4.json'
    if destination.exists():
        raise FileExistsError('Frozen dataset exists; refusing overwrite')
    data = {split: [materialize(task) for task in tasks] for split, tasks in parameters().items()}
    report = {'generation_max_steps': 8, 'interaction_max_turns': 10,
              'holdout_policy': 'Keep independently sampled geometry, disclose overlap; no adaptive rejection or test-driven filtering.',
              'holdout_dataset_rng': 20261015, 'reference_test_previously_seen': True,
              'drop_last': True, 'sampler_order': 'local Python Random(42), not upstream DataLoader order',
              'splits': {}, 'geometry_overlap': {}}
    for split, tasks in data.items():
        lengths = [t['shortest_path'] for t in tasks]
        if any(n is None for n in lengths):
            raise ValueError('Unreachable generated map')
        report['splits'][split] = dict(count=len(tasks), mean_bfs=float(np.mean(lengths)), max_bfs=max(lengths), above_10=sum(n > 10 for n in lengths))
        for task in tasks:
            env = FrozenLakeEnv(size=task['size'], p=task['p'], seed=task['seed'], desc=task['desc'], max_steps=10)
            env.reset()
            if normalize_desc(env.desc) != task['desc']:
                raise AssertionError('Reset changed frozen map')
    for i, a in enumerate(data):
        for b in list(data)[i+1:]:
            report['geometry_overlap'][a + ':' + b] = len({t['map_sha256'] for t in data[a]} & {t['map_sha256'] for t in data[b]})
    changed = 0
    for task in data['test']:
        old, _ = generate_random_map(size=task['size'], p=task['p'], seed=task['seed'], max_steps=10, generation_mode='legacy')
        changed += normalize_desc(old) != task['desc']
    report['test_maps_changed_vs_generation_10'] = changed
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n'
    report['dataset_sha256'] = hashlib.sha256(content.encode()).hexdigest()
    destination.write_text(content)
    (ROOT / 'reports/data_acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
