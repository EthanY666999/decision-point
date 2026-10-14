"""Audit saved evaluations without generating samples or changing training."""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patr_core import ROOT, atomic_json, make_env
from rllm.agents.frozenlake_agent import FrozenLakeAgent

RUNTIME_KEYS = {'micro_batch_size', 'micro_batch_token_limit',
                'vllm_memory_fraction', 'vllm_max_sequences', 'vllm_enforce_eager',
                'vllm_attention_backend'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_evaluation(path, tasks, cfg, split):
    """Recompute aggregates and replay every recorded environment action."""
    metrics = json.loads(path.read_text())
    records = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
    expected = {task['uid']: task for task in tasks}
    require(len(expected) == 100, 'Expected 100 distinct evaluation task IDs')
    require(len(records) == 400, 'Expected 400 evaluation trajectories')
    groups = defaultdict(list)
    parser = FrozenLakeAgent(max_steps=cfg['max_turns'])
    actions = []
    for record in records:
        uid = record['task']['uid']
        require(record['task'] == expected.get(uid), f'Wrong task or split: {uid}')
        groups[uid].append(record)
        require(0 < record['response_tokens'] <= cfg['max_response_tokens'], 'Invalid token count')
        require(record['terminal'] in {'success', 'hole', 'turn_limit', 'token_limit', 'context_limit'},
                'Evaluation contains a tree or unknown terminal state')
        env = make_env(record['task'], cfg)
        reward = 0.0
        turns = 0
        done = False
        for item in record['trace']:
            if item['action'] is None:
                require(item.get('terminal') == 'token_limit', 'Unexpected unexecuted response')
                continue
            require(not done, 'Action recorded after terminal environment state')
            parsed = int(parser._parse_model_response(item['response'])[1])
            require(parsed == item['action'], 'Recorded action differs from upstream parser')
            observation, earned, done, _ = env.step(item['action'])
            require(item['state'] == int(env.s), 'Environment replay state mismatch')
            require(item['observation'] == observation, 'Environment replay observation mismatch')
            require(item['reward'] == earned, 'Environment replay reward mismatch')
            reward += earned
            turns += 1
            actions.append(item['action'])
        require(turns == record['turns'] and turns <= cfg['max_turns'], 'Turn count mismatch')
        require(reward == record['reward'], 'Trajectory outcome reward mismatch')
        terminal = record['terminal']
        require((terminal == 'success') == bool(env.success()), 'Success terminal mismatch')
        if terminal == 'hole':
            require(done and not env.success(), 'Hole terminal mismatch')
        elif terminal == 'turn_limit':
            require(not done and turns == cfg['max_turns'], 'Turn-limit terminal mismatch')
    require(set(groups) == set(expected), 'Evaluation task coverage mismatch')
    for uid, group in groups.items():
        require(len(group) == 4 and {r['lineage'] for r in group} == {'0', '1', '2', '3'},
                f'Expected four independent root samples for {uid}')
    recomputed = dict(
        success_rate_pct=100 * sum(r['reward'] > 0 for r in records) / 400,
        pass_at_4_pct=100 * sum(any(r['reward'] > 0 for r in group) for group in groups.values()) / 100,
        response_length=sum(r['response_tokens'] for r in records) / 400,
        turns=sum(r['turns'] for r in records) / 400,
        invalid_action_fraction=(sum(a == 0 for a in actions) / len(actions) if actions else None),
        n_tasks=100, n_trajectories=400,
        terminal_counts=dict(Counter(r['terminal'] for r in records)),
        generated_tokens=sum(r['response_tokens'] for r in records), split=split)
    for key, expected_value in recomputed.items():
        value = metrics.get(key)
        equal = (isinstance(value, (float, int)) and math.isfinite(value)
                 and math.isclose(value, expected_value, rel_tol=1e-10, abs_tol=1e-10)) if isinstance(expected_value, float) else value == expected_value
        require(equal, f'Metric mismatch for {key}: saved={value}, recomputed={expected_value}')
    return dict(path=str(path.relative_to(ROOT)), status='verified', metrics=recomputed,
                trajectories_sha256=hashlib.sha256(path.with_suffix('.jsonl').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-validation', action='store_true')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'configs/frozenlake_05b.json').read_text())
    data_path = ROOT / 'artifacts/data.json'
    data = json.loads(data_path.read_text())
    manifest = json.loads((ROOT / 'artifacts/data_manifest.json').read_text())
    require(hashlib.sha256(data_path.read_bytes()).hexdigest() == manifest['sha256'], 'Data manifest mismatch')
    results = []
    validations = []
    for method in cfg['methods']:
        run = ROOT / 'runs/formal' / f'{method}_seed{cfg["seed"]}'
        status = json.loads((run / 'status.json').read_text()) if (run / 'status.json').exists() else {}
        if args.include_validation:
            for path in sorted(run.glob('validation_*.json')):
                validations.append(audit_evaluation(path, data['validation'], cfg, 'validation'))
        if status.get('status') != 'complete':
            results.append(dict(method=method, status='pending', training_status=status.get('status', 'not_started')))
            continue
        require(status['step'] == cfg['steps'] == 400, f'{method}: incomplete training')
        run_cfg = json.loads((run / 'config.json').read_text())
        require(run_cfg.get('method') == method, 'Wrong method in run configuration')
        require({k: v for k, v in run_cfg.items() if k not in RUNTIME_KEYS | {'method'}} ==
                {k: v for k, v in cfg.items() if k not in RUNTIME_KEYS}, 'Frozen configuration changed')
        history = {r['step']: r for r in map(json.loads, (run / 'train_metrics.jsonl').read_text().splitlines())}
        require(set(history) == set(range(1, 401)), f'{method}: missing training iterations')
        require(all(math.isfinite(r[k]) for r in history.values() for k in ('loss', 'kl', 'grad_norm')),
                f'{method}: nonfinite training statistics')
        require((run / 'checkpoint.pt').is_file(), f'{method}: missing checkpoint')
        import torch
        checkpoint = torch.load(run / 'checkpoint.pt', map_location='cpu', mmap=True, weights_only=False)
        require(checkpoint['step'] == 400, f'{method}: final checkpoint is not at step 400')
        require({k: v for k, v in checkpoint['config'].items() if k not in RUNTIME_KEYS} ==
                {k: v for k, v in cfg.items() if k not in RUNTIME_KEYS}, 'Checkpoint configuration changed')
        require(bool(checkpoint['optimizer']['state']), f'{method}: missing optimizer state')
        optimizer_updates = {int(v['step'].item()) for v in checkpoint['optimizer']['state'].values()}
        expected_updates = cfg['steps'] * math.ceil(cfg['train_batch_size'] / cfg['mini_batch_tasks'])
        require(optimizer_updates == {expected_updates},
                f'{method}: optimizer counters {optimizer_updates} do not match {expected_updates} updates')
        del checkpoint
        require(any((run / 'final_model').glob('*.safetensors')), f'{method}: missing final model')
        result = audit_evaluation(run / 'final_test.json', data['test'], cfg, 'test')
        result['method'] = method
        result['training_steps'] = cfg['steps']
        result['optimizer_updates'] = expected_updates
        results.append(result)
    complete = all(r['status'] == 'verified' for r in results)
    report = dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  all_five_verified=complete, formal=results, validation=validations)
    atomic_json(ROOT / 'reports/evaluation_audit.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_complete and not complete:
        raise SystemExit('Formal evaluation is still incomplete; no final reproduction claim allowed')


if __name__ == '__main__':
    main()
