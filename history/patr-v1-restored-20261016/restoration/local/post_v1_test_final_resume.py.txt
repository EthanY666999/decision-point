"""A step-400 checkpoint must respect the window and retain the full test."""
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import train as training


@pytest.mark.parametrize('expired,stop,expect_test', [
    (True, False, False), (False, True, False), (False, False, True),
])
def test_final_checkpoint_resume_obeys_window(tmp_path, monkeypatch, expired, stop, expect_test):
    cfg = json.loads((training.ROOT / 'configs/frozenlake_05b.json').read_text())
    for directory in ('configs', 'artifacts', 'runs/formal/grpo_seed42'):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / 'configs/frozenlake_05b.json').write_text(json.dumps(cfg))
    (tmp_path / 'artifacts/deadline.json').write_text(json.dumps({'unix': 0 if expired else 1e12}))
    for name in ('train.py', 'patr_core.py'):
        (tmp_path / name).write_text('# test source snapshot\n')
    run = tmp_path / 'runs/formal/grpo_seed42'
    (run / 'checkpoint.pt').touch()
    state = dict(config=cfg, model={}, optimizer={}, step=400,
                 rng=np.random.default_rng(42).bit_generator.state,
                 torch_rng=torch.get_rng_state(), cuda_rng=[],
                 python_rng=random.getstate(), elapsed_seconds=10.)
    actor = SimpleNamespace(parameters=lambda: [], load_state_dict=lambda _: None,
                            save_pretrained=lambda *args, **kwargs: None)
    tokenizer = SimpleNamespace(save_pretrained=lambda *args, **kwargs: None)
    optimizer = SimpleNamespace(load_state_dict=lambda _: None)
    evaluations = []
    checkpoints = []

    def evaluate(engine, tokenizer, cfg, tasks, path, seed, split):
        evaluations.append((len(tasks), cfg['eval_samples'], seed, split))
        return {'n_tasks': len(tasks), 'n_trajectories': len(tasks) * cfg['eval_samples']}

    monkeypatch.setattr(training, 'ROOT', tmp_path)
    monkeypatch.setattr(training, 'STOP', stop)
    monkeypatch.setattr(training, 'build_models', lambda *args: (actor, None, None, tokenizer))
    monkeypatch.setattr(training, 'make_datasets', lambda: {'validation': list(range(100)), 'test': list(range(100))})
    monkeypatch.setattr(training, 'sync_weights', lambda *args: None)
    monkeypatch.setattr(training, 'evaluate', evaluate)
    monkeypatch.setattr(training, 'save_checkpoint', lambda cp, actor, opt, step, *args: checkpoints.append(step))
    monkeypatch.setattr(training.torch, 'load', lambda *args, **kwargs: state)
    monkeypatch.setattr(training.torch.optim, 'AdamW', lambda *args, **kwargs: optimizer)
    monkeypatch.setattr(training.torch.cuda, 'set_rng_state_all', lambda _: None)
    monkeypatch.setattr(training.signal, 'signal', lambda *args: None)
    monkeypatch.setenv('PATR_PROGRESS_PATH', str(run / 'progress.json'))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--method', 'grpo', '--resume'])
    training.main()
    status = json.loads((run / 'status.json').read_text())
    assert status['step'] == 400
    if expect_test:
        assert evaluations == [(100, 4, 200042, 'test')]
        assert status['status'] == 'complete'
    else:
        assert evaluations == []
        assert checkpoints == [400]
        assert status['status'] == 'paused_checkpointed'
        assert status['final_evaluation_pending'] is True


def test_first_update_is_checkpointed_before_second_rollout_failure(tmp_path, monkeypatch):
    cfg = json.loads((training.ROOT / 'configs/frozenlake_05b.json').read_text())
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/frozenlake_05b.json').write_text(json.dumps(cfg))
    for name in ('train.py', 'patr_core.py'):
        (tmp_path / name).write_text('# test source snapshot\n')
    checkpoints = []
    calls = []

    def collect(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError('Token id 151725 is out of vocabulary')
        return [], {}, []

    actor = SimpleNamespace(parameters=lambda: [])
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    monkeypatch.setattr(training, 'STOP', False)
    monkeypatch.setattr(training, 'build_models', lambda *args: (actor, None, None, None))
    monkeypatch.setattr(training, 'make_datasets', lambda: {'validation': list(range(100)), 'train': list(range(100)), 'test': list(range(100))})
    monkeypatch.setattr(training, 'evaluate', lambda *args: {})
    monkeypatch.setattr(training, 'collect', collect)
    monkeypatch.setattr(training, 'update', lambda *args: {})
    monkeypatch.setattr(training, 'sync_weights', lambda *args: None)
    monkeypatch.setattr(training, 'write_rollouts', lambda *args: None)
    monkeypatch.setattr(training, 'save_checkpoint', lambda cp, actor, opt, step, *args: checkpoints.append(step))
    monkeypatch.setattr(training.torch.optim, 'AdamW', lambda *args, **kwargs: None)
    monkeypatch.setattr(training.torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(training.signal, 'signal', lambda *args: None)
    monkeypatch.setenv('PATR_PROGRESS_PATH', str(tmp_path / 'progress.json'))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--method', 'tree_random', '--resume'])
    with pytest.raises(ValueError, match='151725'):
        training.main()
    assert checkpoints == [1]
    assert len(calls) == 2
