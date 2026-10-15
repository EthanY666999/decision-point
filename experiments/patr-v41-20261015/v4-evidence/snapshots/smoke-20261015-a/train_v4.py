"""Gated independent GRPO smoke and PaTR/GRPO training driver.

Run only from an accepted frozen snapshot. A failed run does not retry itself.
All resumed work is recorded in a new attempt directory, retaining prior evidence.
"""
import os
os.environ['C_INCLUDE_PATH'] = '/home/vipuser/patr-reproduction/artifacts/build-deps/usr/include/python3.12:/home/vipuser/patr-reproduction/artifacts/build-deps/usr/include'
import argparse
import gzip
import hashlib
import json
import shutil
import signal
import time
from pathlib import Path
import numpy as np
import torch
from engine_v4 import build_models, sync_weights, inference_cache_boundary
from patr_core import collect, evaluation_metrics, atomic_json, append_json
from training_contract import EpochSampler, Counters
from update_v4 import update
from r1_diagnostics import fingerprint
from run_integrity import sha256_file
import checkpoint_v4

ROOT = Path(__file__).resolve().parent
STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def write_rollouts(path, branches, tasks):
    with gzip.open(path, 'wt') as stream:
        for b in branches:
            stream.write(json.dumps(dict(task_uid=tasks[b.task_id]['uid'], task_id=b.task_id,
                lineage=b.lineage, reward=b.reward, terminal=b.terminal, actions=b.actions,
                ids=b.ids, mask=b.mask, trace=b.trace), allow_nan=False)+'\n')


def evaluate(engine, tokenizer, cfg, tasks, directory, step, split, identity):
    if STOP or time.time()>=cfg['deadline_unix']-900:
        raise TimeoutError('Stop/deadline protection before evaluation')
    result_path = directory/f'{split}_{step}.json'
    if result_path.exists():
        previous=json.loads(result_path.read_text())
        if previous['fingerprint_sha256']!=identity['sha256'] or previous['step']!=step:
            raise ValueError('Saved evaluation identity mismatch')
        # Only reuse an atomically completed metric with its retained raw evidence.
        if not Path(previous['raw_path']).is_file():
            raise ValueError('Saved evaluation raw evidence missing')
        if sha256_file(Path(previous['raw_path']))!=previous['raw_sha256']:
            raise ValueError('Saved evaluation raw evidence changed')
        return previous
    inference_cache_boundary(engine,'before_evaluation',reset=True)
    tick=time.time()
    branches, stats, _=collect(tasks,engine,tokenizer,cfg,'grpo',np.random.default_rng(np.random.SeedSequence([42,step,100])),evaluation=True)
    inference_cache_boundary(engine,'after_evaluation',reset=True)
    raw=directory/f'{split}_{step}.jsonl.gz'
    if raw.exists():
        raw.rename(directory/f'{split}_{step}.incomplete-{time.time_ns()}.jsonl.gz')
    write_rollouts(raw,branches,tasks)
    result=dict(step=step,split=split,metrics=evaluation_metrics(branches,tasks),cost=stats,
        seconds=time.time()-tick,raw_path=str(raw),raw_sha256=sha256_file(raw),fingerprint_sha256=identity['sha256'])
    atomic_json(result_path,result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','patr','grpo'],required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--r1-summary',required=True)
    parser.add_argument('--smoke-summary')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config_candidate.json').read_text())
    if time.time()>=cfg['deadline_unix']-900:
        raise TimeoutError('Deadline protection before model initialization')
    if shutil.disk_usage(ROOT).free<20*1024**3:
        raise OSError('Less than 20GiB before training startup')
    identity=fingerprint(cfg)
    frozen=json.loads((ROOT/'frozen_identity.json').read_text())
    if identity!=frozen:
        raise ValueError('Frozen source/config/model/data/dependency drift')
    gate=json.loads((ROOT/'reports/r0_gate.json').read_text())
    if not gate.get('passed') or gate['fingerprint_sha256']!=identity['sha256']:
        raise ValueError('Missing R0 gate for actual snapshot')
    r1=json.loads(Path(args.r1_summary).read_text())
    if not r1.get('completed') or r1.get('optimizer_updates')!=0:
        raise ValueError('R1 is not complete')
    r1_identity=json.loads((Path(args.r1_summary).parent/'fingerprint.json').read_text())
    for key in ('complete_config','data_sha256','model_files','runtime_execution','numerical_environment'):
        if r1_identity[key]!=identity[key]:
            raise ValueError('R1 protocol/assets differ: '+key)
    for name in ('patr_core.py','engine_v4.py','tokenization_v4.py','numerical.py','training_contract.py','trajectory_budget.py'):
        if r1_identity['sources'][name]!=identity['sources'][name]:
            raise ValueError('R1 rollout implementation differs: '+name)
    for key,value in dict(optimizer_minibatch=256,response_width=10240,micro_batch_size=2,micro_batch_token_limit=6000,clip_low=.2,clip_high=.28,ppo_epochs=1).items():
        if cfg[key]!=value:
            raise ValueError('Implementation/config mismatch: '+key)
    if args.phase!='smoke':
        if not args.smoke_summary:
            raise ValueError('Independent GRPO smoke acceptance required')
        smoke=json.loads(Path(args.smoke_summary).read_text())
        if not smoke.get('completed') or not smoke.get('restore_verified') or smoke['implementation_fingerprint_sha256']!=identity['sha256']:
            raise ValueError('Smoke acceptance does not match this implementation')
    implementation_identity=identity
    identity=dict(identity,run_phase=args.phase,run_method='grpo' if args.phase=='smoke' else args.phase)
    identity['sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=Path(args.out)
    if out.exists()!=args.resume:
        raise ValueError('Use a new output path or explicit same-fingerprint resume')
    out.mkdir(parents=True,exist_ok=True)
    if args.resume and json.loads((out/'fingerprint.json').read_text())!=identity:
        raise ValueError('Run fingerprint mismatch')
    atomic_json(out/'fingerprint.json',identity)
    if not (out/'resource_recoveries.json').exists():
        atomic_json(out/'resource_recoveries.json',dict(total=0,ram_original_restart=0,cuda_original_restart=0,history=[]))
    attempt=out/f'attempt-{time.time_ns()}'
    attempt.mkdir()
    eval_dir=out/'evaluations';eval_dir.mkdir(exist_ok=True)
    atomic_json(attempt/'start.json',dict(resume=args.resume,phase=args.phase,time=time.time()))
    data=json.loads((ROOT/'artifacts/data_v4.json').read_text())
    sampler=EpochSampler(len(data['train']))
    counts=Counters()
    method='grpo' if args.phase=='smoke' else args.phase
    limit=2 if args.phase=='smoke' else 400
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    actor,reference,engine,tokenizer=build_models(cfg,42)
    optimizer=torch.optim.AdamW(actor.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'],weight_decay=cfg['weight_decay'])
    cp=out/'latest.pt'
    if args.resume:
        checkpoint_v4.restore(cp,actor,optimizer,sampler,counts,identity)
    else:
        checkpoint_v4.save(cp,actor,optimizer,sampler,counts,identity)
    sync_weights(actor,engine)
    tick=time.time()
    attempt_initial_step=counts.accepted_iteration
    baseline_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if args.phase!='smoke' and counts.accepted_iteration==0:
        evaluate(engine,tokenizer,cfg,data['validation'],eval_dir,0,'validation',identity)
    try:
        while counts.accepted_iteration<limit:
            if STOP or time.time()>=cfg['deadline_unix']-900:
                checkpoint_v4.save(cp,actor,optimizer,sampler,counts,identity)
                atomic_json(out/'status.json',dict(state='paused_saved',counts=counts.state_dict(),reason='signal_or_deadline'))
                return
            if shutil.disk_usage(out).free<20*1024**3:
                checkpoint_v4.save(cp,actor,optimizer,sampler,counts,identity)
                raise OSError('Below 20GiB free; saved at safe boundary')
            if fingerprint(cfg)!=implementation_identity:
                raise ValueError('Frozen identity changed before batch')
            index=sampler.next_batch();tasks=[data['train'][i] for i in index]
            batch=counts.raw_batch_index
            os.environ['PATR_PROGRESS_PATH']=str(out/'progress.json')
            atomic_json(out/'status.json',dict(state='running',phase=args.phase,counts=counts.state_dict(),epoch=sampler.epoch,pid=os.getpid(),updated_unix=time.time()))
            inference_cache_boundary(engine,'before_training_rollout',reset=True)
            started=time.time()
            branches,stats,events=collect(tasks,engine,tokenizer,cfg,method,np.random.default_rng(np.random.SeedSequence([42,batch])),evaluation=False)
            inference_cache_boundary(engine,'after_training_rollout',reset=True)
            write_rollouts(attempt/f'raw_{batch}.jsonl.gz',branches,tasks)
            before=actor.model.layers[0].self_attn.q_proj.weight.detach().clone()
            result=update(actor,reference,optimizer,branches,tokenizer,counts,len(tasks))
            accepted=bool(result['retained_trajectories'])
            can_continue=counts.record_batch(len(tasks),stats['generated_tokens'],accepted)
            parameter_change=float((actor.model.layers[0].self_attn.q_proj.weight-before).abs().max())
            if result['optimizer_updates'] and not parameter_change:
                raise RuntimeError('Optimizer reports update but monitored parameter unchanged')
            record=dict(step=counts.accepted_iteration,raw_batch=batch,epoch=sampler.epoch,
                task_indices=index,rollout=stats,optimization=result,counts=counts.state_dict(),
                parameter_change=parameter_change,seconds=time.time()-started,events=events,
                free_bytes=shutil.disk_usage(out).free,peak_cuda_bytes=torch.cuda.max_memory_allocated())
            append_json(attempt/'metrics.jsonl',record)
            if not can_continue:
                checkpoint_v4.save(cp,actor,optimizer,sampler,counts,identity)
                raise RuntimeError('20 consecutive empty accepted batches; resource protection')
            if not accepted:continue
            sync_weights(actor,engine)
            step=counts.accepted_iteration
            if step%5==0 or args.phase=='smoke' or step==40:
                checkpoint_v4.save(cp,actor,optimizer,sampler,counts,identity)
            if step in (5,10) or step%40==0:
                # Projections are measured estimates, never a promise to finish.
                elapsed=time.time()-tick
                steps_this_attempt=step-attempt_initial_step
                evidence_bytes=sum(p.stat().st_size for p in attempt.rglob('*') if p.is_file())
                remaining=max(0,limit-step)
                projected_evidence=evidence_bytes*remaining/max(1,steps_this_attempt)
                required=projected_evidence+6*1024**3+2*1024**3+4*1024**3
                forecast=dict(step=step,elapsed_seconds_this_attempt=elapsed,
                    remaining_seconds_mid=elapsed*remaining/max(1,steps_this_attempt),
                    uncertainty='Early measured extrapolation; filtering, lengths, evaluation and interruptions can change cost',
                    additional_evidence_bytes_estimate=projected_evidence,
                    reserve_including_atomic_checkpoint_final_model_safety=required,
                    free_bytes=shutil.disk_usage(out).free,deadline_unix=cfg['deadline_unix'])
                atomic_json(out/f'forecast_{step}.json',forecast)
                if shutil.disk_usage(out).free<max(20*1024**3,required):
                    checkpoint_v4.save(cp,actor,optimizer,sampler,counts,identity)
                    raise OSError('Measured remaining evidence/checkpoint capacity budget not met')
            if args.phase!='smoke' and step in cfg['eval_steps']:
                evaluate(engine,tokenizer,cfg,data['validation'],eval_dir,step,'validation',identity)
            if step==40 and args.phase!='smoke':
                # In-place same-fingerprint restore validates the engineering checkpoint.
                checkpoint_v4.restore(cp,actor,optimizer,sampler,counts,identity)
                sync_weights(actor,engine)
                atomic_json(out/'step40_acceptance.json',dict(passed=True,step=40,fingerprint_sha256=identity['sha256']))
        checkpoint_v4.save(cp,actor,optimizer,sampler,counts,identity)
        checkpoint_v4.restore(cp,actor,optimizer,sampler,counts,identity)
        sync_weights(actor,engine)
        if args.phase!='smoke':
            evaluate(engine,tokenizer,cfg,data['test'],eval_dir,400,'reference_test',identity)
            evaluate(engine,tokenizer,cfg,data['holdout'],eval_dir,400,'holdout',identity)
            actor.save_pretrained(out/'final_model',safe_serialization=True)
            tokenizer.save_pretrained(out/'final_model')
        current_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        summary=dict(completed=True,phase=args.phase,restore_verified=True,counts=counts.state_dict(),seconds_this_attempt=time.time()-tick,bytes_growth=current_bytes-baseline_bytes,fingerprint_sha256=identity['sha256'],implementation_fingerprint_sha256=implementation_identity['sha256'])
        atomic_json(out/'summary.json',summary)
        atomic_json(out/'status.json',dict(state='complete',**summary))
    except Exception as exc:
        # Never overwrite a known checkpoint with a possibly partial/nonfinite update.
        atomic_json(out/'status.json',dict(state='paused_error',exception=repr(exc),counts_in_memory=counts.state_dict(),checkpoint=str(cp),updated_unix=time.time()))
        raise
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
