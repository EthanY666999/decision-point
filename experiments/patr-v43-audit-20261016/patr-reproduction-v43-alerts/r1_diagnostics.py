"""Fixed 3x64 training-task tree diagnostics; no optimizer and no formal test."""
import os
os.environ['C_INCLUDE_PATH'] = '/home/vipuser/patr-reproduction/artifacts/build-deps/usr/include/python3.12:/home/vipuser/patr-reproduction/artifacts/build-deps/usr/include'
import argparse
import gzip
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from engine_v4 import build_models, sync_weights, inference_cache_boundary
from patr_core import collect, atomic_json, append_json
from training_contract import EpochSampler
from run_integrity import build_fingerprint

ROOT = Path(__file__).resolve().parent


def fingerprint(cfg):
    result = build_fingerprint(cfg, ROOT/'artifacts/data_v4.json', root=ROOT)
    # Old helper excludes runtime knobs; V4 freezes the complete configuration too.
    result['complete_config'] = cfg
    result['sha256'] = hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('R1 output exists; diagnose before any restart')
    cfg = json.loads((ROOT/'config_candidate.json').read_text())
    if time.time() >= cfg['deadline_unix'] - 900:
        raise RuntimeError('Deadline save window reached')
    if shutil.disk_usage(ROOT).free < 20*1024**3:
        raise RuntimeError('Less than 20GiB free')
    expected = json.loads((ROOT/'frozen_identity.json').read_text())
    actual = fingerprint(cfg)
    if actual != expected:
        raise ValueError('Frozen R1 source/config/data/model/dependency identity changed')
    gate = json.loads((ROOT/'reports/r0_gate.json').read_text())
    if not gate.get('passed') or gate['fingerprint_sha256'] != actual['sha256']:
        raise ValueError('No valid R0 gate for this source snapshot')
    out.mkdir(parents=True)
    atomic_json(out/'fingerprint.json', actual)
    data = json.loads((ROOT/'artifacts/data_v4.json').read_text())
    sampler = EpochSampler(len(data['train']))
    indices = [sampler.next_batch() for _ in range(3)]
    atomic_json(out/'task_indices.json',indices)
    start = time.time()
    actor, reference, engine, tokenizer = build_models(cfg,42)
    sync_weights(actor,engine)
    records = []
    for method in ('grpo','patr'):
        for batch_index, index in enumerate(indices):
            if time.time() >= cfg['deadline_unix']-900:
                raise RuntimeError('Deadline protection before next diagnostic batch')
            tasks = [data['train'][i] for i in index]
            atomic_json(out/'status.json',dict(stage='R1',method=method,batch=batch_index,state='running',pid=os.getpid(),updated_unix=time.time()))
            os.environ['PATR_PROGRESS_PATH']=str(out/'progress.json')
            inference_cache_boundary(engine,'before_r1_batch',reset=True)
            tick=time.time()
            # Same predeclared request-seed input for both methods in each batch.
            rng=np.random.default_rng(np.random.SeedSequence([42,batch_index]))
            branches, stats, events=collect(tasks,engine,tokenizer,cfg,method,rng,evaluation=False)
            inference_cache_boundary(engine,'after_r1_batch',reset=True)
            groups=[[b for b in branches if b.task_id==i] for i in range(64)]
            zero=sum(all(b.reward==0 for b in group) for group in groups)
            one=sum(all(b.reward==1 for b in group) for group in groups)
            actions=[a for b in branches for a in b.actions]
            traces=[t for b in branches for t in b.trace]
            # Expansion scores are the top-k, and scores are nonnegative.
            route_groups={}
            for event in events:
                if event['kind']=='expand':
                    route_groups.setdefault((event['task_id'],event['turn']),[]).append(event['score'])
            record=dict(method=method,batch=batch_index,seconds=time.time()-tick,stats=stats,
                leaves_per_task=[len(g) for g in groups],outcome_all_zero_groups=zero,outcome_all_one_groups=one,
                learning_zero_variance_groups=zero+one,signal_groups=64-zero-one,
                signal_trajectories=sum(len(g) for g in groups if len({b.reward for b in g})>1),
                forked_task_fraction=len({e['task_id'] for e in events if e['kind']=='expand'})/64,
                route_barriers=len(route_groups),all_zero_score_barriers=sum(all(s==0 for s in scores) for scores in route_groups.values()),
                event_counts=dict(Counter(e['kind'] for e in events)),direction_counts=dict(Counter(actions)),
                valid_noop_fraction=sum(t['valid_noop'] for t in traces)/len(traces),
                canonical_assistant_tokens=sum(b.ntokens for b in branches),
                post_prompt_tokens=sum(b.budget.counted_tokens for b in branches),
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),free_disk_bytes=shutil.disk_usage(out).free)
            with gzip.open(out/f'{method}_{batch_index}_evidence.jsonl.gz','wt') as stream:
                for b in branches:
                    stream.write(json.dumps(dict(task_uid=tasks[b.task_id]['uid'],lineage=b.lineage,reward=b.reward,terminal=b.terminal,trace=b.trace))+'\n')
                stream.write(json.dumps(dict(events=events))+'\n')
            append_json(out/'metrics.jsonl',record)
            records.append(record)
    # Recheck source identity before calling the stage complete.
    if fingerprint(cfg) != expected:
        raise ValueError('Identity drift during R1')
    atomic_json(out/'status.json',dict(stage='R1',state='complete',batches=len(records),seconds=time.time()-start,updated_unix=time.time()))
    atomic_json(out/'summary.json',dict(completed=True,optimizer_updates=0,batches=records,seconds=time.time()-start,fingerprint_sha256=expected['sha256']))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        # Never auto-retry a failed diagnostic or falsely mark completion.
        print(json.dumps(dict(state='failed',exception=repr(exc))),flush=True)
        raise
