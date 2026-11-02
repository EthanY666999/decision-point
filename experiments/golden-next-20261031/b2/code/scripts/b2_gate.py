"""Predeclared B2 decision. Reads training/validation only, never final_test."""
import argparse
import json
import math
from pathlib import Path


def assess(patr, random):
    for arm in (patr, random):
        rows = arm['metrics']
        assert [r['step'] for r in rows] == list(range(1, 151)), 'Both arms must complete150'
        assert all(math.isfinite(r[k]) for r in rows for k in ['loss', 'grad_norm', 'max_parameter_change', 'mixed_group_fraction'])
        assert all(0 <= r['mixed_group_fraction'] <= 1 for r in rows)
        assert all(r['max_leaves'] <= 8 for r in rows)
        assert set(arm['validation']) == {100, 125, 150}
        assert all(math.isfinite(x) for x in arm['validation'].values())
    deltas = {step: patr['validation'][step] - random['validation'][step] for step in (100, 125, 150)}
    mixed = {name: sum(r['mixed_group_fraction'] for r in arm['metrics'][100:150]) / 50
             for name, arm in [('patr', patr), ('tree_random', random)]}
    return dict(pass_gate=all(x > 0 for x in deltas.values()) and mixed['patr'] > mixed['tree_random'],
                validation_success_delta_pp=deltas, mixed_fraction_steps101_150=mixed,
                training_validation_only=True, test_used=False,
                gate='Strictly higher PaTR validation Success at100/125/150 AND higher mixed-group mean101--150.')


def read_run(root, method):
    state = json.loads((root / 'status.json').read_text())
    assert state['status'] == 'complete' and state['step'] == state['total_steps'] == 150
    assert state['method'] == method
    identity = json.loads((root / 'b2_identity.json').read_text())
    assert not identity['smoke'] and identity['method'] == method
    cfg = identity['config']
    assert cfg['steps'] == 150 and cfg['eval_every'] == 25 and cfg['branch_interval'] == 2 and cfg['leaf_cap'] == 8
    metrics = [json.loads(l) for l in (root / 'train_metrics.jsonl').read_text().splitlines()]
    validation = {step: json.loads((root / f'validation_{step:04d}.json').read_text())['success_rate_pct'] for step in (100, 125, 150)}
    return dict(metrics=metrics, validation=validation), identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patr-run', type=Path, required=True)
    parser.add_argument('--random-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    p, pi = read_run(args.patr_run, 'patr')
    r, ri = read_run(args.random_run, 'tree_random')
    assert pi['config'] == ri['config'] and pi['source_sha256'] == ri['source_sha256'] and pi['protocol'] == ri['protocol']
    result = assess(p, r)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
