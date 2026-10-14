"""CPU failure-order regressions for vLLM idle/cache-reset guards."""
from types import SimpleNamespace

import pytest
import torch
import train


def fake_engine(unfinished=0, used=0, waiting=0, reset=True):
    calls=dict(load=0,verify=0,reset=0,synchronize=0,collect=0)
    allocator=SimpleNamespace(get_num_total_blocks=lambda:10,get_num_free_blocks=lambda:10-used)
    manager=SimpleNamespace(block_tables={},cross_block_tables={},
        block_allocator=SimpleNamespace(_allocators={'GPU':allocator}))
    scheduler=SimpleNamespace(waiting=[object()]*waiting,running=[],swapped=[],_async_stopped=[],block_manager=manager)
    backend=SimpleNamespace(unfinished=unfinished,scheduler=[scheduler])
    backend.get_num_unfinished_requests=lambda:backend.unfinished
    backend.has_unfinished_requests=lambda:bool(backend.unfinished)
    def reset_cache():
        calls['reset']+=1
        return reset
    backend.reset_prefix_cache=reset_cache
    def load(values):
        calls['load']+=1
        return [name for name,value in values]
    backend.model_executor=SimpleNamespace(driver_worker=SimpleNamespace(
        model_runner=SimpleNamespace(model=SimpleNamespace(load_weights=load))))
    return SimpleNamespace(llm_engine=backend),calls


def sync_mocks(monkeypatch,calls):
    monkeypatch.setattr(train,'verify_qwen_weights',lambda *args:calls.__setitem__('verify',calls['verify']+1))
    monkeypatch.setattr(train.torch.cuda,'synchronize',lambda:calls.__setitem__('synchronize',calls['synchronize']+1))
    return torch.nn.Linear(1,1,bias=False)


@pytest.mark.parametrize('options',[{'unfinished':1},{'waiting':1},{'used':1}])
def test_sync_refuses_busy_engine_before_weight_write(monkeypatch,options):
    engine,calls=fake_engine(**options)
    actor=sync_mocks(monkeypatch,calls)
    with pytest.raises(RuntimeError,match='not idle'):
        train.sync_weights(actor,engine)
    assert calls==dict(load=0,verify=0,reset=0,synchronize=0,collect=0)


def test_sync_failed_reset_aborts_before_successful_return(monkeypatch):
    engine,calls=fake_engine(reset=False)
    actor=sync_mocks(monkeypatch,calls)
    with pytest.raises(RuntimeError,match='Prefix-cache reset failed'):
        train.sync_weights(actor,engine)
    assert calls==dict(load=1,verify=1,reset=1,synchronize=0,collect=0)


def test_sync_idle_success_keeps_weight_count_interface(monkeypatch):
    engine,calls=fake_engine()
    actor=sync_mocks(monkeypatch,calls)
    assert train.sync_weights(actor,engine)==1
    assert calls==dict(load=1,verify=1,reset=1,synchronize=1,collect=0)


def evaluation_mocks(monkeypatch,engine,calls,leave_pending=False):
    def collect(*args,**kwargs):
        calls['collect']+=1
        if leave_pending:engine.llm_engine.unfinished=1
        return [],{'generated_tokens':0},[]
    monkeypatch.setattr(train,'collect',collect)
    monkeypatch.setattr(train,'evaluation_metrics',lambda *args:{'n_tasks':0})


def test_evaluation_busy_refusal_preserves_existing_output(tmp_path,monkeypatch):
    engine,calls=fake_engine(unfinished=1)
    evaluation_mocks(monkeypatch,engine,calls)
    output=tmp_path/'validation.json';output.write_bytes(b'original evidence')
    with pytest.raises(RuntimeError,match='not idle'):
        train.evaluate(engine,SimpleNamespace(eos_token_id=1),{},[],output,100042,'validation')
    assert output.read_bytes()==b'original evidence' and calls['collect']==0 and calls['reset']==0


def test_evaluation_checks_idle_again_before_writing_results(tmp_path,monkeypatch):
    engine,calls=fake_engine()
    evaluation_mocks(monkeypatch,engine,calls,leave_pending=True)
    output=tmp_path/'validation.json'
    with pytest.raises(RuntimeError,match='not idle'):
        train.evaluate(engine,SimpleNamespace(eos_token_id=1),{},[],output,100042,'validation')
    assert calls['collect']==1 and not output.exists() and not output.with_suffix('.jsonl').exists()


def test_evaluation_success_keeps_saved_metrics_and_records(tmp_path,monkeypatch):
    engine,calls=fake_engine()
    evaluation_mocks(monkeypatch,engine,calls)
    output=tmp_path/'validation.json'
    result=train.evaluate(engine,SimpleNamespace(eos_token_id=1),{},[],output,100042,'validation')
    assert result['n_tasks']==0 and result['eos_token_id']==1
    assert output.is_file() and output.with_suffix('.jsonl').read_bytes()==b''
    assert calls['reset']==2


def test_evaluation_reset_failure_aborts_before_sampling(tmp_path,monkeypatch):
    engine,calls=fake_engine(reset=False)
    evaluation_mocks(monkeypatch,engine,calls)
    output=tmp_path/'validation.json'
    with pytest.raises(RuntimeError,match='Prefix-cache reset failed'):
        train.evaluate(engine,SimpleNamespace(eos_token_id=1),{},[],output,100042,'validation')
    assert calls['reset']==1 and calls['collect']==0 and not output.exists()


def test_exit_reset_occurs_after_sampling_before_output_or_return(tmp_path,monkeypatch):
    engine,calls=fake_engine()
    evaluation_mocks(monkeypatch,engine,calls)
    events=[]
    original_collect=train.collect
    def collect(*args,**kwargs):
        events.append('collect');return original_collect(*args,**kwargs)
    def reset():
        calls['reset']+=1;events.append('reset');return True
    monkeypatch.setattr(train,'collect',collect)
    engine.llm_engine.reset_prefix_cache=reset
    output=tmp_path/'validation.json'
    train.evaluate(engine,SimpleNamespace(eos_token_id=1),{},[],output,100042,'validation')
    assert events==['reset','collect','reset']
    assert calls['reset']==2 and output.exists()


def test_exit_reset_failure_preserves_prior_outputs_and_aborts(tmp_path,monkeypatch):
    engine,calls=fake_engine()
    evaluation_mocks(monkeypatch,engine,calls)
    def reset():
        calls['reset']+=1;return calls['reset']==1
    engine.llm_engine.reset_prefix_cache=reset
    output=tmp_path/'validation.json';output.write_bytes(b'prior metrics')
    raw=output.with_suffix('.jsonl');raw.write_bytes(b'prior raw')
    with pytest.raises(RuntimeError,match='Prefix-cache reset failed'):
        train.evaluate(engine,SimpleNamespace(eos_token_id=1),{},[],output,100042,'validation')
    assert calls['collect']==1 and calls['reset']==2
    assert output.read_bytes()==b'prior metrics' and raw.read_bytes()==b'prior raw'
