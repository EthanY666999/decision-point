"""CPU regression of real DAPO main-loop aggregation across candidate batches.

Model and sampling fixtures avoid GPU execution while preserving main control flow.
"""
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_final_resume import short_fixture
import train as training


@pytest.mark.parametrize('candidate_batches', [2, 3])
def test_dapo_main_weights_all_candidate_diagnostics(tmp_path, monkeypatch, candidate_batches):
    cfg, _ = short_fixture(tmp_path, monkeypatch)
    cfg.update(train_batch_size=4, active_method='dapo')
    (tmp_path / 'configs/short_test.json').write_text(json.dumps(cfg))
    candidates = []
    candidate_task_ids = []
    calls = []

    def collect(tasks, *args):
        attempt = len(calls)
        calls.append(attempt)
        candidate_task_ids.extend(t['uid'] for t in tasks)
        if attempt >= candidate_batches:
            pytest.fail('Expected candidate groups should already fill the training batch')
        per_task = [2, 6, 4][attempt]
        branches = []
        for ti in range(len(tasks)):
            mixed = attempt > 0 and (candidate_batches == 2 or ti < 2)
            invalid = not mixed
            for index in range(per_task):
                reward = float(mixed and index % 2)
                branch = SimpleNamespace(task_id=ti, reward=reward,
                    training_reward=reward - 0.05 * invalid, ntokens=10 + ti, turns=1,
                    invalid=invalid, terminal='success' if reward else 'turn_limit')
                branches.append(branch)
        candidates.extend(branches)
        episodes = len(branches)
        stats = dict(episodes=episodes, success_rate=sum(b.reward > 0 for b in branches) / episodes,
            mean_training_reward=sum(b.training_reward for b in branches) / episodes,
            first_turn_invalid_fraction=sum(b.invalid for b in branches) / episodes,
            mean_response_tokens=sum(b.ntokens for b in branches) / episodes, mean_turns=1.0,
            generated_tokens=sum(b.ntokens for b in branches), input_tokens=100 * episodes,
            executed_actions=episodes, invalid_actions=sum(b.invalid for b in branches),
            terminal_counts=dict(Counter(b.terminal for b in branches)))
        stats['invalid_action_fraction'] = stats['invalid_actions'] / episodes
        return branches, stats, []

    def update(actor, ref, optimizer, branches, cfg, method, tokenizer, rng, n_tasks):
        assert method == 'dapo' and n_tasks == 4
        assert {b.task_id for b in branches} == {0, 1, 2, 3}
        return dict(loss=0.0, kl=0.0, grad_norm=0.0, optimizer_updates=1)

    monkeypatch.setattr(training, 'STOP', False)
    monkeypatch.setattr(training, 'build_models', lambda *args: (SimpleNamespace(parameters=lambda: []), None, None, None))
    monkeypatch.setattr(training, 'evaluate', lambda *args: {})
    monkeypatch.setattr(training, 'collect', collect)
    monkeypatch.setattr(training, 'update', update)
    monkeypatch.setattr(training, 'sync_weights', lambda *args: None)
    monkeypatch.setattr(training, 'write_rollouts', lambda *args: None)
    monkeypatch.setattr(training, 'save_checkpoint', lambda path,*args: path.write_text('fixture checkpoint'))
    monkeypatch.setattr(training.torch.optim, 'AdamW', lambda *args, **kwargs: None)
    monkeypatch.setattr(training.torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setenv('PATR_PROGRESS_PATH', str(tmp_path / 'progress.json'))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--method', 'dapo', '--config',
        'configs/short_test.json', '--stop-after-step', '1'])
    training.main()
    run = tmp_path / 'runs/short_validation/dapo_seed42'
    records = [json.loads(line) for line in (run / 'train_metrics.jsonl').read_text().splitlines()]
    assert len(records) == 1 and len(calls) == candidate_batches
    assert len(set(candidate_task_ids)) == 4 * candidate_batches
    record = records[0]
    assert record['resample_batches'] == candidate_batches
    assert record['accepted_distinct_task_uids'] == 4
    assert record['accepted_success_rate'] == 0.5
    assert record['episodes'] == len(candidates)
    assert record['generated_tokens'] == sum(b.ntokens for b in candidates)
    assert record['invalid_actions'] == sum(b.invalid for b in candidates)
    assert record['success_rate'] == pytest.approx(sum(b.reward > 0 for b in candidates) / len(candidates))
    expected = dict(mean_training_reward=sum(b.training_reward for b in candidates) / len(candidates),
                    first_turn_invalid_fraction=sum(b.invalid for b in candidates) / len(candidates))
    assert {key: record[key] for key in expected} == pytest.approx(expected)
