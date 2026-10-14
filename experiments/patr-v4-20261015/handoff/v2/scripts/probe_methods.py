"""Five-method one-step GPU integration probe; never launches formal training.

The optimizer replay updates are disposable diagnostics. Each method's reported
evaluation is restored to its first sampled-batch update, not a replay endpoint.
DAPO rejection sampling mirrors train.main but is an independent copy, explicitly
not evidence that the CLI main loop has been integration-tested.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train as tr  # Sets vLLM environment before importing GPU dependencies.
import numpy as np
import torch
from numerical import verify_qwen_weights
from patr_core import collect, atomic_json, make_datasets
from protocol_v2 import profile_config
from run_integrity import build_fingerprint


def learning_reward(branch):
    value = getattr(branch, 'training_reward', None)
    return branch.reward if value is None else value


def digest_state(value):
    """Stable value digest; GPU tensors are copied one at a time, not retained."""
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            y = x.detach().cpu().contiguous()
            h.update(str((str(y.dtype), tuple(y.shape))).encode())
            h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=repr):
                add(key); add(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for item in x: add(item)
        else:
            h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def restore_checkpoint(path, actor, optimizer, rng):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    actor.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    rng.bit_generator.state = saved['rng']
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    random.setstate(saved['python_rng'])
    del saved
    gc.collect()


def rng_digest(rng):
    return digest_state(dict(numpy=rng.bit_generator.state, python=random.getstate(),
                             torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))


def random_draws(rng):
    return dict(numpy=rng.integers(0,2**31,8).tolist(), python=[random.random() for _ in range(8)],
                torch=torch.rand(8).tolist(), cuda=torch.rand(8,device='cuda').cpu().tolist())


def checkpoint_probe(out, actor, ref, optimizer, branches, cfg, method, tokenizer, rng, n_tasks):
    path = out/'temporary_checkpoint.pt'
    expected = dict(model=digest_state(actor.state_dict()), optimizer=digest_state(optimizer.state_dict()),
                    rng=rng_digest(rng))
    tr.save_checkpoint(path, actor, optimizer, 1, rng, cfg, 0.)
    report = dict(checkpoint_sha256=file_sha256(path), checkpoint_bytes=path.stat().st_size,
                  optimizer_state_entries=len(optimizer.state), baseline_state_sha256=expected,
                  retained_training_steps=1, discarded_replay_sampled_batches=2,
                  discarded_replay_optimizer_updates_per_batch=int(np.ceil(n_tasks/cfg['mini_batch_tasks'])))
    wanted_draws = random_draws(rng)
    # Ensure the test restores changed state instead of comparing an untouched object.
    with torch.no_grad():
        next(actor.parameters()).view(-1)[0].add_(1.)
        first_state = next(iter(optimizer.state.values()))
        first_state['exp_avg'].view(-1)[0].add_(1.)
    random_draws(rng)
    restore_checkpoint(path, actor, optimizer, rng)
    actual = dict(model=digest_state(actor.state_dict()), optimizer=digest_state(optimizer.state_dict()),
                  rng=rng_digest(rng))
    report['roundtrip_exact'] = {key: actual[key] == expected[key] for key in expected}
    report['next_random_draws_exact'] = random_draws(rng) == wanted_draws
    replay = []
    for repetition in (1,2):
        restore_checkpoint(path, actor, optimizer, rng)
        fixed_rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'],1,1702]))
        metrics = tr.update(actor,ref,optimizer,branches,cfg,method,tokenizer,fixed_rng,n_tasks)
        replay.append(dict(repetition=repetition, model_sha256=digest_state(actor.state_dict()),
                           optimizer_sha256=digest_state(optimizer.state_dict()),
                           rng_sha256=rng_digest(rng), optimizer_rng_sha256=digest_state(fixed_rng.bit_generator.state),
                           metrics=metrics))
        atomic_json(out/'checkpoint_probe_in_progress.json', dict(report, replay=replay))
    report['fixed_batch_replay'] = replay
    report['fixed_batch_replay_exact'] = {
        key: replay[0][key] == replay[1][key]
        for key in ('model_sha256','optimizer_sha256','rng_sha256','optimizer_rng_sha256','metrics')}
    # Only the actual one-step model goes into subsequent synchronization/evaluation.
    restore_checkpoint(path, actor, optimizer, rng)
    report['one_step_model_restored'] = digest_state(actor.state_dict()) == expected['model']
    path.unlink()
    report['temporary_checkpoint_removed'] = not path.exists()
    atomic_json(out/'checkpoint_probe.json', report)
    return report


def sample_batch(data, engine, tokenizer, cfg, method, rng, out):
    """Mirror train.main task selection and unique-candidate DAPO filtering."""
    task_rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'],1,1701]))
    indices = task_rng.choice(len(data['train']),cfg['train_batch_size'],replace=False).tolist()
    used = set(indices)
    accepted, accepted_tasks, accepted_uids = [], [], set()
    candidate_stats, all_events = [], []
    for attempt in range(1,cfg['dapo_max_resample_batches']+1):
        tasks = [data['train'][i] for i in indices]
        branches,stats,events = collect(tasks,engine,tokenizer,cfg,method,rng)
        candidate_stats.append(dict(attempt=attempt,task_indices=indices,task_uids=[t['uid'] for t in tasks],stats=stats))
        all_events.extend(dict(event=e, attempt=attempt, task_uid=tasks[e['task_id']]['uid']) for e in events)
        atomic_json(out/'candidate_stats.json', candidate_stats)
        atomic_json(out/'branch_events.json', all_events)
        if method != 'dapo':
            return branches,tasks,dict(raw_batches=candidate_stats,resample_batches=1),all_events
        tr.write_rollouts(out/'dapo_candidates.jsonl.gz',branches,tasks,'wt' if attempt==1 else 'at',attempt)
        groups = {ti:[b for b in branches if b.task_id == ti] for ti in range(len(tasks))}
        for ti,task in enumerate(tasks):
            group = groups[ti]
            if (len({learning_reward(b) for b in group}) > 1 and task['uid'] not in accepted_uids
                    and len(accepted_tasks)<cfg['train_batch_size']):
                for b in group: b.task_id=len(accepted_tasks)
                accepted.extend(group); accepted_tasks.append(task); accepted_uids.add(task['uid'])
        if len(accepted_tasks) == cfg['train_batch_size']:
            assert len(accepted_uids) == len(accepted_tasks)
            return accepted,accepted_tasks,dict(raw_batches=candidate_stats,resample_batches=attempt,
                unique_candidate_task_indices=len(used),accepted_distinct_task_uids=len(accepted_uids),
                filtering_target='training_reward', implementation='independent mirror of train.main, not CLI integration'),all_events
        if attempt >= cfg['dapo_max_resample_batches']:
            break
        remaining = [i for i in range(len(data['train'])) if i not in used]
        if len(remaining)<cfg['train_batch_size']:
            raise RuntimeError('DAPO exhausted unique candidate tasks')
        indices = task_rng.choice(remaining,cfg['train_batch_size'],replace=False).tolist()
        used.update(indices)
    raise RuntimeError('DAPO dynamic sampling could not fill batch; no silent fallback')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/frozenlake_05b.json')
    parser.add_argument('--output',default='runs/method_probe_v2')
    parser.add_argument('--methods',nargs='+',default=['grpo','patr','tree_random','dapo','arpo'],
                        choices=['grpo','patr','tree_random','dapo','arpo'])
    parser.add_argument('--train-tasks',type=int,choices=[2,3,4],default=4)
    args = parser.parse_args()
    out = (ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT): raise ValueError('Output must stay in v2 checkout')
    out.mkdir(parents=True,exist_ok=False)
    cfg = json.loads((ROOT/args.config).read_text())
    override_path = ROOT/'artifacts/runtime_override.json'
    if override_path.exists():
        override = json.loads(override_path.read_text())
        if not set(override) <= tr.OVERRIDE_KEYS:
            raise ValueError('Runtime overrides contain non-engineering settings')
        cfg.update(override)
    cfg.update(profile_config('behavior_variant_v2'))
    cfg.update(run_mode='short_validation',steps=1,train_batch_size=args.train_tasks,mini_batch_tasks=2,
               micro_batch_size=2,micro_batch_token_limit=6000,validation_tasks=4,
               dataset_path='artifacts/data_v2.json', project_forward_bf16=True,
               recompute_old_logprobs=True,kl_estimator='categorical',rollout_is_max=2.0,
               diagnostic_gradient_split=True,vllm_enforce_eager=True,eval_every=1,save_every=1,
               max_response_tokens=10240,max_context_tokens=32768,eval_samples=4,
               protocol_note='One-step diagnostics, not a paper replication or formal training run')
    atomic_json(out/'config.json',cfg)
    atomic_json(out/'fingerprint.json',build_fingerprint(cfg,ROOT/cfg['dataset_path'],root=ROOT))
    data = make_datasets(cfg)
    # Never select, inspect or evaluate reserved test/holdout rows.
    validation = data['validation'][:4]
    assert len(validation)==4 and all(t.get('split')=='validation' for t in validation)
    atomic_json(out/'status.json',dict(status='initializing',methods=args.methods))
    actor,ref,engine,tokenizer = tr.build_models(cfg,cfg['seed'])
    results = {}
    for method in args.methods:
        method_out = out/method; method_out.mkdir()
        cfg = dict(cfg,active_method=method,arpo_vocabulary_policy='tokenizer_vocab')
        cfg['sampling_vocabulary'] = tr.configure_vocabulary(actor,ref,tokenizer,cfg,method)
        atomic_json(method_out/'config.json',cfg)
        atomic_json(method_out/'fingerprint.json',build_fingerprint(cfg,ROOT/cfg['dataset_path'],root=ROOT))
        os.environ['PATR_PROGRESS_PATH'] = str(method_out/'progress.json')
        start = time.monotonic()
        atomic_json(out/'status.json',dict(status='running',method=method,completed=list(results)))
        with torch.no_grad():
            ref_parameters = dict(ref.named_parameters())
            for name,p in actor.named_parameters(): p.copy_(ref_parameters[name].float())
        actor.zero_grad(set_to_none=True); actor.eval()
        torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed']); random.seed(cfg['seed'])
        rng = np.random.default_rng(cfg['seed'])
        optimizer = torch.optim.AdamW(actor.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
        tr.sync_weights(actor,engine)
        initial_sync = verify_qwen_weights(actor,engine)
        initial = tr.evaluate(engine,tokenizer,cfg,validation,method_out/'validation_0000.json',100042,'validation')
        branches,tasks,sampling,events = sample_batch(data,engine,tokenizer,cfg,method,rng,method_out)
        tr.write_rollouts(method_out/'rollouts_0001.jsonl.gz',branches,tasks)
        atomic_json(method_out/'training_tasks.json',tasks)
        optimizer_rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'],1,1702]))
        changes = tr.update(actor,ref,optimizer,branches,cfg,method,tokenizer,optimizer_rng,len(tasks))
        tr.sync_weights(actor,engine)
        after_sync = verify_qwen_weights(actor,engine)
        atomic_json(method_out/'one_step_update.json',changes)
        restoration = checkpoint_probe(method_out,actor,ref,optimizer,branches,cfg,method,tokenizer,rng,len(tasks))
        tr.sync_weights(actor,engine)
        restored_sync = verify_qwen_weights(actor,engine)
        final = tr.evaluate(engine,tokenizer,cfg,validation,method_out/'validation_0001.json',100042,'validation')
        record = dict(method=method,seed=cfg['seed'],retained_sampled_batches=1,
            n_train_tasks=len(tasks),n_trajectories=len(branches),sampling=sampling,update=changes,
            nonzero_parameter_delta=changes['max_parameter_change']>0,
            initial_sync=initial_sync,after_update_sync=after_sync,restored_sync=restored_sync,
            checkpoint=restoration,initial_validation=initial,final_validation=final,
            seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            limitation='Four validation tasks are only a runtime/behavior smoke, not a statistically reliable improvement estimate')
        record['passed'] = (record['nonzero_parameter_delta']
            and all(s['all_equal'] and s['checked_modules']==171 for s in (initial_sync,after_sync,restored_sync))
            and all(restoration['roundtrip_exact'].values()) and restoration['next_random_draws_exact']
            and all(restoration['fixed_batch_replay_exact'].values()) and restoration['one_step_model_restored'])
        atomic_json(method_out/'result.json',record)
        results[method]=record
        atomic_json(out/'summary.json',dict(passed=all(r['passed'] for r in results.values()),methods=results,
                                           complete=len(results)==len(args.methods)))
        print(json.dumps({'METHOD_PROBE_COMPLETE':method,'update':changes,'final_validation':final}),flush=True)
        del optimizer, branches
        gc.collect(); torch.cuda.empty_cache()
    atomic_json(out/'status.json',dict(status='complete',methods=list(results),formal_training_started=False))
    atomic_json(ROOT/'reports/method_probe.json',dict(passed=all(r['passed'] for r in results.values()),
        completed_methods=list(results),output=str(out),summary=str(out/'summary.json'),
        method_passed={m:r['passed'] for m,r in results.items()},formal_training_started=False))


if __name__=='__main__':
    main()
