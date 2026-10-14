"""Short-run checkpoints respect stop windows and never touch reserved test tasks.

The trainer flow is isolated from model loading here. Real-file provenance and
checkpoint transactions are covered independently by test_run_integrity.py.
"""
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import train as training


def short_fixture(tmp_path, monkeypatch):
    cfg = json.loads((training.ROOT / 'configs/frozenlake_05b.json').read_text())
    cfg.update(steps=20, run_mode='short_validation', validation_tasks=16, active_method='grpo')
    for directory in ('configs', 'artifacts'):
        (tmp_path / directory).mkdir()
    (tmp_path / 'configs/short_test.json').write_text(json.dumps(cfg))
    data = {split: [dict(uid=f'{split}-{i}') for i in range(100)]
            for split in ('validation', 'test', 'train')}
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    monkeypatch.setattr(training, 'make_datasets', lambda cfg: data)
    monkeypatch.setattr(training, 'configure_vocabulary', lambda *args, **kwargs: {'policy': 'mocked-in-flow-test'})
    monkeypatch.setattr(training, 'build_fingerprint', lambda *args, **kwargs: {'sha256': 'mocked-in-flow-test'})
    def prepare(run, resume, fingerprint, config, **kwargs):
        run.mkdir(parents=True, exist_ok=True)
        return {'start_step': 20 if resume else 0, 'recovery': None}
    monkeypatch.setattr(training, 'prepare_run', prepare)
    monkeypatch.setattr(training.signal, 'signal', lambda *args: None)
    return cfg, data


@pytest.mark.parametrize('expired,stop,expect_validation', [
    (True, False, False), (False, True, False), (False, False, True),
])
def test_final_checkpoint_resume_obeys_window(tmp_path, monkeypatch, expired, stop, expect_validation):
    cfg, data = short_fixture(tmp_path, monkeypatch)
    (tmp_path / 'artifacts/deadline.json').write_text(json.dumps({'unix': 0 if expired else 1e12}))
    run = tmp_path / 'runs/short_validation/grpo_seed42'
    run.mkdir(parents=True)
    (run / 'checkpoint.pt').touch()
    state = dict(config=cfg, model={}, optimizer={}, step=20,
                 rng=np.random.default_rng(42).bit_generator.state,
                 torch_rng=torch.get_rng_state(), cuda_rng=[],
                 python_rng=random.getstate(), elapsed_seconds=10.)
    actor = SimpleNamespace(parameters=lambda: [], load_state_dict=lambda _: None)
    tokenizer = SimpleNamespace()
    optimizer = SimpleNamespace(load_state_dict=lambda _: None)
    evaluations, checkpoints = [], []

    def evaluate(engine, tokenizer, cfg, tasks, path, seed, split):
        assert path.name == 'final_validation.json'
        assert tasks == data['validation'][:16]
        assert not any(task in data['test'] for task in tasks)
        evaluations.append((len(tasks), cfg['eval_samples'], seed, split))
        return {'n_tasks': len(tasks), 'n_trajectories': len(tasks) * cfg['eval_samples']}

    monkeypatch.setattr(training, 'STOP', stop)
    monkeypatch.setattr(training, 'build_models', lambda *args: (actor, None, None, tokenizer))
    monkeypatch.setattr(training, 'sync_weights', lambda *args: None)
    monkeypatch.setattr(training, 'evaluate', evaluate)
    monkeypatch.setattr(training, 'save_checkpoint', lambda cp, actor, opt, step, *args: (checkpoints.append(step),cp.touch()))
    monkeypatch.setattr(training.torch, 'load', lambda *args, **kwargs: state)
    monkeypatch.setattr(training.torch.optim, 'AdamW', lambda *args, **kwargs: optimizer)
    monkeypatch.setattr(training.torch.cuda, 'set_rng_state_all', lambda _: None)
    monkeypatch.setenv('PATR_PROGRESS_PATH', str(run / 'progress.json'))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--method', 'grpo', '--config', 'configs/short_test.json', '--resume'])
    training.main()
    status = json.loads((run / 'status.json').read_text())
    assert status['step'] == 20
    if expect_validation:
        assert evaluations == [(16, 4, 100042, 'short_validation')]
        assert status['status'] == 'complete'
        assert status['run_mode'] == 'short_validation'
    else:
        assert evaluations == []
        assert checkpoints == [20]
        assert status['status'] == 'paused_checkpointed'
        assert status['final_evaluation_pending'] is True
    assert not (tmp_path / 'runs/formal').exists()


def test_first_update_is_checkpointed_before_second_rollout_failure(tmp_path, monkeypatch):
    short_fixture(tmp_path, monkeypatch)
    checkpoints, calls = [], []
    def collect(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError('Token id 151725 is out of vocabulary')
        return [], {}, []
    actor = SimpleNamespace(parameters=lambda: [])
    monkeypatch.setattr(training, 'STOP', False)
    monkeypatch.setattr(training, 'build_models', lambda *args: (actor, None, None, None))
    monkeypatch.setattr(training, 'evaluate', lambda *args: {})
    monkeypatch.setattr(training, 'collect', collect)
    monkeypatch.setattr(training, 'update', lambda *args: {})
    monkeypatch.setattr(training, 'sync_weights', lambda *args: None)
    monkeypatch.setattr(training, 'write_rollouts', lambda *args: None)
    monkeypatch.setattr(training, 'save_checkpoint', lambda cp, actor, opt, step, *args: (checkpoints.append(step),cp.touch()))
    monkeypatch.setattr(training.torch.optim, 'AdamW', lambda *args, **kwargs: None)
    monkeypatch.setattr(training.torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setenv('PATR_PROGRESS_PATH', str(tmp_path / 'progress.json'))
    # A fresh short run must never use the old queue's implicit --resume.
    monkeypatch.setattr(sys, 'argv', ['train.py', '--method', 'tree_random', '--config', 'configs/short_test.json'])
    with pytest.raises(ValueError, match='151725'):
        training.main()
    assert checkpoints == [0,1]
    assert len(calls) == 2


@pytest.mark.parametrize('mode,steps', [('formal', 20), ('short_validation', 400)])
def test_formal_or_long_run_rejected_before_model_loading(tmp_path, monkeypatch, mode, steps):
    cfg, _ = short_fixture(tmp_path, monkeypatch)
    cfg.update(run_mode=mode, steps=steps)
    (tmp_path / 'configs/short_test.json').write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--method', 'grpo', '--config', 'configs/short_test.json'])
    monkeypatch.setattr(training, 'build_models', lambda *args: pytest.fail('Model loaded before short-run scope validation'))
    with pytest.raises(ValueError, match='only permits explicit short_validation'):
        training.main()
    assert not (tmp_path / 'runs').exists()


def test_resume_and_override_allowlists_are_distinct():
    from run_integrity import RUNTIME_KEYS
    assert training.RUNTIME_KEYS is RUNTIME_KEYS
    assert 'vllm_attention_backend' not in RUNTIME_KEYS
    assert 'vllm_enforce_eager' not in RUNTIME_KEYS
    assert training.OVERRIDE_KEYS == RUNTIME_KEYS | {'vllm_attention_backend', 'vllm_enforce_eager'}
