"""Real main/recovery files; model/GPU operations are replaced by CPU fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_run_integrity as fixtures
import run_integrity as integrity
import train
from scripts import report,verify_final as verify


class Decoder:
    def decode(self,ids,**kwargs):
        assert kwargs==dict(skip_special_tokens=True,clean_up_tokenization_spaces=False)
        return '```Right```' if ids[0]==10 else '```Left```'


def files_under(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def main_run(tmp_path,monkeypatch):
    evidence=fixtures.RawEvaluationEvidenceTests();evidence.setUp()
    initial_cfg=json.loads((ROOT/'configs/short_legacy.json').read_text())
    cfg=dict(initial_cfg,**evidence.cfg)
    cfg.update(steps=20,eval_every=10,save_every=1,train_batch_size=4,mini_batch_tasks=2,
               validation_tasks=1,active_method='grpo')
    (tmp_path/'configs').mkdir();(tmp_path/'artifacts').mkdir()
    config=tmp_path/'configs/fixture.json';config.write_text(json.dumps(cfg))
    data={'validation':[evidence.task],'test':[],
          'train':[dict(evidence.task,uid='train-'+str(i)) for i in range(8)]}
    data_path=tmp_path/cfg['dataset_path'];data_path.write_text(json.dumps(data))
    run=tmp_path/'runs/short_validation/grpo_seed42'
    payload=dict(schema_version=4,sources={},protocol=integrity.frozen_config(dict(cfg,method='grpo')))
    fp=dict(payload,sha256=integrity._digest(payload))
    integrity.prepare_run(run,False,fp,dict(cfg,method='grpo'),root=tmp_path)
    calls={'evaluations':[],'updates':[],'builds':0,'tokenizer_loads':0}
    opt_state={'state':{},'param_groups':[]}
    def load_opt(state):
        opt_state.clear();opt_state.update(copy.deepcopy(state))
    optimizer=SimpleNamespace(load_state_dict=load_opt,state_dict=lambda:opt_state)
    actor=SimpleNamespace(parameters=lambda:[],load_state_dict=lambda state:None,state_dict=lambda:{})
    def build(*args):
        calls['builds']+=1
        return actor,None,None,SimpleNamespace(eos_token_id=2)
    def write_eval(path,split='validation'):
        value=dict(evidence.metrics,split=split,seconds=1.25)
        path.write_text(json.dumps(value))
        path.with_suffix('.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in evidence.records))
        return value
    def evaluate(engine,tokenizer,actual_cfg,tasks,path,seed,split):
        assert tasks==data['validation'] and seed==100042
        calls['evaluations'].append(path.name)
        return write_eval(path,split)
    def collect(tasks,*args):
        branches=[SimpleNamespace(task_id=i,reward=1.,training_reward=1.) for i in range(len(tasks))]
        return branches,{'generated_tokens':12,'input_tokens':20},[]
    def update(*args):
        calls['updates'].append(1)
        opt_state['state'].setdefault(0,{'step':torch.tensor(0.)})['step']+=2
        return dict(loss=0.1,kl=0.01,grad_norm=0.2,optimizer_updates=2)
    monkeypatch.setattr(train,'ROOT',tmp_path)
    monkeypatch.setattr(train,'build_fingerprint',lambda *args,**kwargs:fp)
    monkeypatch.setattr(train,'make_datasets',lambda cfg:data)
    monkeypatch.setattr(train,'build_models',build)
    monkeypatch.setattr(train,'configure_vocabulary',lambda *args:{'fixture':True})
    monkeypatch.setattr(train,'sync_weights',lambda *args:None)
    monkeypatch.setattr(train,'evaluate',evaluate)
    monkeypatch.setattr(train,'collect',collect)
    monkeypatch.setattr(train,'update',update)
    monkeypatch.setattr(train,'write_rollouts',lambda path,*args,**kwargs:path.write_text('fixture rollout'))
    monkeypatch.setattr(train.torch.optim,'AdamW',lambda *args,**kwargs:optimizer)
    monkeypatch.setattr(train.torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(train.torch.cuda,'set_rng_state_all',lambda state:None)
    monkeypatch.setattr(train.torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(train.signal,'signal',lambda *args:None)
    monkeypatch.setattr(train,'STOP',False)
    def load_verified_tokenizer(actual_cfg,actual_fp):
        assert actual_cfg==cfg and actual_fp==fp
        calls['tokenizer_loads']+=1
        return Decoder(),{'fixture_assets':True}
    monkeypatch.setattr(verify,'load_verified_tokenizer',load_verified_tokenizer)
    monkeypatch.setattr(verify,'incremental_response_text',lambda tokenizer,prompt,ids:
        tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False))
    def checkpoint(step):
        if step:
            opt_state['state'][0]={'step':torch.tensor(float(step*2))}
        state=dict(step=step,config=cfg,model={},optimizer=copy.deepcopy(opt_state),
            rng=np.random.default_rng(42).bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=[],python_rng=random.getstate(),elapsed_seconds=step*1.)
        integrity.atomic_checkpoint(run/'checkpoint.pt',state)
        (run/'status.json').write_text(json.dumps({'status':'paused_checkpointed','step':step}))
        for name in ('train_metrics.jsonl','branch_events.jsonl'):
            (run/name).write_text(''.join(json.dumps(dict(step=i,loss=.1,kl=.01,grad_norm=.2,
                generated_tokens=12,input_tokens=20,step_seconds=1.,events=[]))+'\n' for i in range(1,step+1)))
        for prior in (0,10):
            if prior<step:write_eval(run/f'validation_{prior:04d}.json')
        args=['train.py','--method','grpo','--config','configs/fixture.json','--resume']
        if step<20:args+=['--stop-after-step',str(step+1)]
        monkeypatch.setattr(sys,'argv',args)
    yield SimpleNamespace(root=tmp_path,run=run,cfg=cfg,fp=fp,calls=calls,checkpoint=checkpoint,
        write_eval=write_eval,opt_state=opt_state)
    evidence.doCleanups()


def damage_pair(f,path,kind):
    f.write_eval(path)
    if kind=='partial_metrics':path.write_bytes(b'{"seconds":')
    elif kind=='partial_raw':path.with_suffix('.jsonl').write_bytes(b'{"task":')
    elif kind=='missing_metrics':path.unlink()
    elif kind=='missing_raw':path.with_suffix('.jsonl').unlink()
    elif kind=='missing_both':path.unlink();path.with_suffix('.jsonl').unlink()


@pytest.mark.parametrize('step',[0,10,20])
@pytest.mark.parametrize('kind',['complete','partial_metrics','partial_raw','missing_metrics','missing_raw','missing_both'])
def test_main_resumes_current_scheduled_boundary_before_any_new_update(main_run,step,kind):
    f=main_run;f.checkpoint(step)
    boundary=f.run/f'validation_{step:04d}.json';damage_pair(f,boundary,kind)
    prior={p.name:p.read_bytes() for p in (boundary,boundary.with_suffix('.jsonl')) if p.exists()}
    cp_before=(f.run/'checkpoint.pt').read_bytes()
    committed_before={name:(f.run/name).read_bytes() for name in ('train_metrics.jsonl','branch_events.jsonl')}
    train.main()
    assert f.calls['evaluations'][0]==boundary.name
    assert f.calls['updates']==([] if step==20 else [1])
    assert int(f.opt_state['state'][0]['step'])==2*(step if step==20 else step+1)
    assert f.calls['evaluations']==([boundary.name,'final_validation.json'] if step==20 else [boundary.name])
    status=json.loads((f.run/'status.json').read_text())
    assert status['status']==('complete' if step==20 else 'paused_checkpointed')
    train.audit_saved_validation(boundary,[json.loads(boundary.with_suffix('.jsonl').read_text().splitlines()[0])['task']],f.cfg,Decoder())
    assert f.calls['tokenizer_loads']==int(step>0)
    archives=list((f.run/'recovery_history').glob('*/incident.json'))
    assert len(archives)==bool(prior)
    if prior:
        archive=archives[0].parent
        assert {name:(archive/name).read_bytes() for name in prior}==prior
    for name,before in committed_before.items():
        assert (f.run/name).read_bytes().startswith(before)
    if step==20:assert (f.run/'checkpoint.pt').read_bytes()==cp_before
    attempts=list((f.run/'recovery_evaluations').glob('*.json'))
    assert len(attempts)==1
    cost=json.loads(attempts[0].read_text())
    assert cost['status']=='complete' and cost['validation_step']==step
    assert cost['recorded_metric_seconds']==1.25 and cost['measured_wall_seconds']>0


@pytest.mark.parametrize('earlier',[0,10])
@pytest.mark.parametrize('kind',['partial_metrics','partial_raw','missing_metrics','missing_raw','missing_both'])
def test_missing_or_damaged_earlier_eval_refuses_before_any_evidence_write(main_run,earlier,kind):
    f=main_run;f.checkpoint(20)
    f.write_eval(f.run/'validation_0020.json')
    f.write_eval(f.run/'final_validation.json','short_validation')
    damage_pair(f,f.run/f'validation_{earlier:04d}.json',kind)
    before=files_under(f.root)
    with pytest.raises(ValueError,match='Earlier scheduled validation'):
        train.main()
    assert files_under(f.root)==before and f.calls['builds']==0 and not f.calls['updates']


@pytest.mark.parametrize('earlier',[0,10])
def test_earlier_eval_same_length_ids_trace_consistent_but_text_forged_refuses_before_run_write(main_run,earlier):
    f=main_run;f.checkpoint(20)
    path=f.run/f'validation_{earlier:04d}.json'
    raw=path.with_suffix('.jsonl')
    rows=[json.loads(line) for line in raw.read_text().splitlines()]
    rows[0]['ids'][1]=12
    rows[0]['trace'][0]['response_token_ids'][0]=12
    raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    # Structural/environment-only checks pass: the unchanged saved action is
    # Right, although these equally long IDs actually decode to Left.
    assert verify.audit_evaluation(path,[rows[0]['task']],f.cfg,'validation')['status']=='verified'
    before=files_under(f.run)
    with pytest.raises(ValueError,match='Earlier scheduled validation') as error:
        train.main()
    assert 'decode to different saved text' in str(error.value.__cause__)
    assert files_under(f.run)==before and f.calls['builds']==0 and not f.calls['updates']
    assert f.calls['tokenizer_loads']==1


def test_resume_tokenizer_asset_failure_precedes_any_run_evidence_write(main_run,monkeypatch):
    f=main_run;f.checkpoint(20)
    def fail(*args):
        raise ValueError('Original tokenizer/model manifest changed')
    monkeypatch.setattr(verify,'load_verified_tokenizer',fail)
    before=files_under(f.run)
    with pytest.raises(ValueError,match='Earlier scheduled validation') as error:
        train.main()
    assert 'manifest changed' in str(error.value.__cause__)
    assert files_under(f.run)==before and f.calls['builds']==0 and not f.calls['updates']


def test_boundary_evaluation_failure_cannot_mark_complete_and_cost_is_retained(main_run,monkeypatch):
    f=main_run;f.checkpoint(20)
    before=(f.run/'checkpoint.pt').read_bytes()
    def fail(engine,tokenizer,cfg,tasks,path,seed,split):
        assert path.name=='validation_0020.json'
        path.write_bytes(b'{"cut":')
        raise OSError('fixture partial evaluation write')
    monkeypatch.setattr(train,'evaluate',fail)
    with pytest.raises(OSError,match='partial evaluation'):
        train.main()
    assert json.loads((f.run/'status.json').read_text())['status']!='complete'
    assert not (f.run/'final_validation.json').exists()
    assert not f.calls['updates'] and (f.run/'checkpoint.pt').read_bytes()==before
    attempt=json.loads(next((f.run/'recovery_evaluations').glob('*.json')).read_text())
    assert attempt['status']=='failed' and attempt['measured_wall_seconds']>0
    assert attempt['recorded_metric_seconds'] is None


@pytest.mark.parametrize('stop,expired',[(True,False),(False,True)])
def test_pending_boundary_respects_stop_window_without_rewriting_checkpoint(main_run,monkeypatch,stop,expired):
    f=main_run;f.checkpoint(20)
    monkeypatch.setattr(train,'STOP',stop)
    (f.root/'artifacts/deadline.json').write_text(json.dumps({'unix':0 if expired else 1e12}))
    before=(f.run/'checkpoint.pt').read_bytes()
    train.main()
    status=json.loads((f.run/'status.json').read_text())
    assert status['status']=='paused_checkpointed' and status['validation_pending']==20
    assert not f.calls['evaluations'] and not f.calls['updates']
    assert (f.run/'checkpoint.pt').read_bytes()==before


def test_report_keeps_recorded_discard_cost_and_unknown_retry_cost_separate(main_run):
    f=main_run;f.checkpoint(20)
    f.write_eval(f.run/'validation_0020.json')
    train.main()
    pending=f.run/'recovery_evaluations/unfinished.json'
    pending.write_text('{"status":"started","validation_step":20,"measured_wall_seconds":null}')
    rows,_=report.build_report(f.run.parent,dict(f.cfg,methods=['grpo']))
    row=rows[0]
    assert row['recovery_discarded_evaluation_seconds_recorded']==1.25
    assert row['recovery_retry_evaluation_wall_seconds_measured']>0
    assert row['recovery_retry_evaluation_unmeasured_attempts']==1
    assert row['recovery_evaluation_cost_complete'] is False


def test_earlier_validation_callback_must_explicitly_confirm_verified(main_run):
    f=main_run;f.checkpoint(20)
    before=files_under(f.root)
    with pytest.raises(ValueError,match='Earlier scheduled validation'):
        integrity.prepare_run(f.run,True,f.fp,dict(f.cfg,method='grpo'),root=f.root,
                              evaluation_validator=lambda path:{'status':'pending'})
    assert files_under(f.root)==before


@pytest.mark.parametrize('stop,expired',[(True,False),(False,True)])
def test_regular_boundary_checks_window_again_after_optimizer_and_checkpoint(main_run,monkeypatch,stop,expired):
    f=main_run;f.checkpoint(9)
    original_update=train.update
    def update(*args):
        value=original_update(*args)
        monkeypatch.setattr(train,'STOP',stop)
        (f.root/'artifacts/deadline.json').write_text(json.dumps({'unix':0 if expired else 1e12}))
        return value
    monkeypatch.setattr(train,'update',update)
    train.main()
    assert f.calls['updates']==[1] and f.calls['evaluations']==[]
    status=json.loads((f.run/'status.json').read_text())
    assert status['status']=='paused_checkpointed' and status['step']==status['validation_pending']==10
    checkpoint=torch.load(f.run/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert checkpoint['step']==10 and int(checkpoint['optimizer']['state'][0]['step'])==20
    assert not (f.run/'validation_0010.json').exists()
    # The next invocation repairs step10 before exactly one newly requested step11.
    monkeypatch.setattr(train,'STOP',False);monkeypatch.setattr(train,'update',original_update)
    (f.root/'artifacts/deadline.json').write_text('{"unix":1000000000000}')
    monkeypatch.setattr(sys,'argv',['train.py','--method','grpo','--config','configs/fixture.json','--resume','--stop-after-step','11'])
    train.main()
    assert f.calls['updates']==[1,1] and f.calls['evaluations']==['validation_0010.json']
    assert int(f.opt_state['state'][0]['step'])==22


@pytest.mark.parametrize('stop,expired',[(True,False),(False,True)])
def test_fresh_initial_evaluation_respects_window_and_leaves_resumable_step_zero(main_run,monkeypatch,stop,expired):
    f=main_run
    monkeypatch.setattr(train,'STOP',stop)
    (f.root/'artifacts/deadline.json').write_text(json.dumps({'unix':0 if expired else 1e12}))
    monkeypatch.setattr(sys,'argv',['train.py','--method','grpo','--config','configs/fixture.json','--run-root','runs/fresh'])
    train.main()
    run=f.root/'runs/fresh/grpo_seed42'
    status=json.loads((run/'status.json').read_text())
    assert status['status']=='paused_checkpointed' and status['step']==status['validation_pending']==0
    checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert checkpoint['step']==0 and not f.calls['evaluations'] and not f.calls['updates']
    assert checkpoint['optimizer']['state']=={}
    assert not (run/'validation_0000.json').exists()


def test_fresh_initial_evaluation_failure_keeps_atomic_cp0_and_resume_does_not_recommit_it(main_run,monkeypatch):
    f=main_run;run=f.root/'runs/fresh_failure/grpo_seed42'
    original_evaluate=train.evaluate;original_save=train.save_checkpoint
    saved_steps=[]
    def save(path,actor,opt,step,*args):
        saved_steps.append(step);return original_save(path,actor,opt,step,*args)
    def fail(engine,tokenizer,cfg,tasks,path,seed,split):
        checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert checkpoint['step']==0 and checkpoint['optimizer']['state']=={}
        path.write_bytes(b'{"partial_initial":')
        raise OSError('initial evaluation interrupted')
    monkeypatch.setattr(train,'save_checkpoint',save);monkeypatch.setattr(train,'evaluate',fail)
    args=['train.py','--method','grpo','--config','configs/fixture.json','--run-root','runs/fresh_failure']
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(OSError,match='initial evaluation interrupted'):
        train.main()
    assert saved_steps==[0] and f.calls['updates']==[]
    assert json.loads((run/'status.json').read_text())['status']!='complete'
    initialization=(run/'initial_checkpoint.json').read_bytes()
    cost=json.loads(initialization)
    assert cost['checkpoint_bytes']==(run/'checkpoint.pt').stat().st_size and cost['checkpoint_write_seconds']>0
    partial=(run/'validation_0000.json').read_bytes()
    monkeypatch.setattr(train,'evaluate',original_evaluate)
    monkeypatch.setattr(sys,'argv',args+['--resume','--stop-after-step','1'])
    train.main()
    assert saved_steps.count(0)==1 and f.calls['updates']==[1]
    assert f.calls['evaluations']==['validation_0000.json']
    assert (run/'initial_checkpoint.json').read_bytes()==initialization
    archive=next((run/'recovery_history').glob('*/incident.json')).parent
    assert (archive/'validation_0000.json').read_bytes()==partial
    assert json.loads((run/'status.json').read_text())['step']==1
