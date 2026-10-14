"""Queue regressions: failures/pauses must not launch another formal method."""
import json
from types import SimpleNamespace

import pytest
from scripts import run_queue as q


def run_fake_queue(tmp_path, monkeypatch, outcomes, verification_rc=0, old_log=''):
    for name in ('artifacts', 'configs', 'logs', 'scripts', 'vendor'):
        (tmp_path / name).mkdir()
    cfg = dict(methods=['grpo', 'patr'], seed=42, micro_batch_size=16,
               micro_batch_token_limit=24000, vllm_max_sequences=256)
    (tmp_path / 'configs/frozenlake_05b.json').write_text(json.dumps(cfg))
    (tmp_path / 'logs/formal_grpo.log').write_text(old_log)
    launched = []
    outcomes = iter(outcomes)

    def popen(cmd, cwd, stdout, stderr):
        method = cmd[cmd.index('--method') + 1]
        launched.append(method)
        rc, status, message = next(outcomes)
        run = tmp_path / 'runs/formal' / f'{method}_seed42'
        run.mkdir(parents=True, exist_ok=True)
        (run / 'status.json').write_text(json.dumps(dict(status=status)))
        stdout.write(message)
        stdout.flush()
        return SimpleNamespace(poll=lambda: rc, returncode=rc)

    def execute(cmd, **kwargs):
        return SimpleNamespace(returncode=verification_rc if 'scripts/verify_final.py' in cmd else 0)

    monkeypatch.setattr(q, 'ROOT', tmp_path)
    monkeypatch.setattr(q.subprocess, 'Popen', popen)
    monkeypatch.setattr(q.subprocess, 'run', execute)
    q.main()
    return launched, json.loads((tmp_path / 'artifacts/queue_status.json').read_text())


@pytest.mark.parametrize('rc,status,reason', [
    (1, 'training', 'method_failed'),
    (0, 'paused_checkpointed', 'method_paused_or_incomplete'),
    (0, 'training', 'method_paused_or_incomplete'),
])
def test_incomplete_method_stops_queue(tmp_path, monkeypatch, rc, status, reason):
    launched, queue = run_fake_queue(tmp_path, monkeypatch, [(rc, status, 'CUDA illegal memory access')])
    assert launched == ['grpo']
    assert queue['stop_reason'] == reason
    assert queue['status'] == 'stopped_incomplete'


def test_verified_method_advances(tmp_path, monkeypatch):
    launched, queue = run_fake_queue(tmp_path, monkeypatch, [(0, 'complete', '')] * 2)
    assert launched == ['grpo', 'patr']
    assert queue['status'] == 'complete'


def test_failed_verification_stops_queue(tmp_path, monkeypatch):
    launched, queue = run_fake_queue(tmp_path, monkeypatch, [(0, 'complete', '')], verification_rc=1)
    assert launched == ['grpo']
    assert queue['stop_reason'] == 'final_verification_failed'


def test_old_oom_does_not_retry_new_cuda_fault(tmp_path, monkeypatch):
    launched, queue = run_fake_queue(tmp_path, monkeypatch, [(1, 'training', 'illegal memory access')],
                                     old_log='previous attempt: CUDA out of memory\n')
    assert launched == ['grpo']
    assert queue['stop_reason'] == 'method_failed'
    assert not (tmp_path / 'artifacts/runtime_override.json').exists()


def test_oom_retries_same_method_with_memory_changes_only(tmp_path, monkeypatch):
    launched, queue = run_fake_queue(tmp_path, monkeypatch, [
        (1, 'training', 'CUDA out of memory'), (0, 'complete', ''), (0, 'complete', '')])
    assert launched == ['grpo', 'grpo', 'patr']
    runtime = json.loads((tmp_path / 'artifacts/runtime_override.json').read_text())
    assert runtime == dict(micro_batch_size=8, micro_batch_token_limit=12000, vllm_max_sequences=128)
    assert queue['status'] == 'complete'
