#!/usr/bin/env python3
"""Analyze prespecified validation endpoints, never test/holdout or best checkpoints.

Example:
  python scripts/analyze_short_validation.py \
    --run legacy=runs/short_legacy/grpo_seed42 \
    --run repaired=runs/short_behavior/grpo_seed42 --pair legacy:repaired
Frozen-weight prompt comparisons accept --evaluation label=validation.json.
JSONL trajectories beside each metric JSON are required for paired statistics.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_integrity import atomic_json, read_step_records

ALLOWED_SPLITS = {'validation', 'short_validation'}


def mean(values):
    return float(np.mean(values)) if values else None


def load(path):
    return json.loads(Path(path).read_text())


def finite_numbers(rows, key):
    return [float(r[key]) for r in rows if r.get(key) is not None
            and isinstance(r[key], (int, float)) and math.isfinite(r[key])]


def numeric_summary(values):
    return dict(n=len(values), mean=mean(values), median=statistics.median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None,
                p95=float(np.quantile(values,.95)) if values else None)


def per_group_metrics(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['task']['uid']].append(r)
    traces = [t for r in rows for t in r.get('trace', [])]
    first = [r['trace'][0] for r in rows if r.get('trace')]
    executed = [t for t in traces if t.get('action') is not None]
    valid = lambda t: t.get('action_valid', t.get('action') in (1,2,3,4))
    # Missing EOS counts must remain unknown, never silently mean zero.
    lengths = [sum(r['mask']) if 'mask' in r else r['response_tokens'] for r in rows]
    incl = [t['response_tokens'] for t in traces if 'response_tokens' in t]
    excl = [t['response_tokens_excluding_eos'] for t in traces if 'response_tokens_excluding_eos' in t]
    action_seqs = {uid: {tuple(t['action'] for t in r.get('trace', []) if t.get('action') is not None)
                         for r in rs} for uid,rs in groups.items()}
    response_seqs = {uid: {tuple(t.get('response','') for t in r.get('trace',[])) for r in rs}
                     for uid,rs in groups.items()}
    terminals = Counter(r.get('terminal','unknown') for r in rows)
    return dict(n_tasks=len(groups), n_trajectories=len(rows),
        success_rate_pct=100*mean([r['reward']>0 for r in rows]),
        pass_at_4_pct=100*mean([any(r['reward']>0 for r in rs) for rs in groups.values()]),
        response_length=mean(lengths),
        response_length_excluding_eos=sum(excl)/len(rows) if len(excl)==len(traces) and traces else None,
        response_tokens_per_turn_including_eos=mean(incl) if len(incl)==len(traces) else None,
        response_tokens_per_turn_excluding_eos=mean(excl) if len(excl)==len(traces) else None,
        turns=mean([r['turns'] for r in rows]), sampled_response_turns=len(traces),
        executed_actions=len(executed), invalid_action_fraction=mean([t['action']==0 for t in executed]),
        first_turn_invalid_fraction=mean([not valid(t) for t in first]),
        format_invalid_fraction=mean([not valid(t) for t in traces]),
        first_turn_format_counts=dict(Counter(t.get('format_status','unknown') for t in first)),
        all_responses_invalid_trajectory_fraction=mean([bool(r.get('trace')) and all(not valid(t) for t in r['trace']) for r in rows]),
        valid_noop_fraction=mean([bool(t.get('valid_noop',False)) for t in executed]) if all('valid_noop' in t for t in executed) else None,
        terminal_counts=dict(terminals), hole_fraction=terminals['hole']/len(rows),
        turn_limit_fraction=terminals['turn_limit']/len(rows),
        token_limit_fraction=terminals['token_limit']/len(rows),
        context_limit_fraction=terminals['context_limit']/len(rows),
        unique_action_sequences_per_task=mean([len(s) for s in action_seqs.values()]),
        identical_four_action_sequences_fraction=mean([len(s)==1 for s in action_seqs.values()]),
        unique_response_sequences_per_task=mean([len(s) for s in response_seqs.values()]),
        sampling_seed_missing_count=sum(t.get('sampling_seed') is None for t in traces),
        eos_count_missing_turns=len(traces)-len(excl),
        response_tokens_missing_turns=len(traces)-len(incl))


def read_evaluation(path):
    path = Path(path)
    metrics = load(path)
    if metrics.get('split') not in ALLOWED_SPLITS:
        raise ValueError(f'Refusing non-validation evaluation: {path}, split={metrics.get("split")!r}')
    raw_path = path.with_suffix('.jsonl')
    if not raw_path.exists():
        raise ValueError(f'Paired audit needs raw trajectories: {raw_path}')
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    if not rows:
        raise ValueError(f'Empty evaluation: {raw_path}')
    groups = defaultdict(list)
    for row in rows:
        task = row['task']
        if task.get('split') != 'validation':
            raise ValueError(f'Refusing non-validation raw task {task.get("uid")}: {task.get("split")}')
        groups[task['uid']].append(row)
        if 'mask' in row and row.get('response_tokens',sum(row['mask'])) != sum(row['mask']):
            raise ValueError('Raw response length differs from assistant-token mask')
        if 'mask' in row and 'ids' in row and len(row['mask']) != len(row['ids']):
            raise ValueError('Token ID/mask alignment mismatch')
        traced = [t.get('response_tokens') for t in row.get('trace',[])]
        if traced and all(x is not None for x in traced) and 'mask' in row and sum(traced)!=sum(row['mask']):
            raise ValueError('Trace lengths differ from assistant-token mask')
    for uid, samples in groups.items():
        if len(samples)!=4 or len({r['lineage'] for r in samples})!=4:
            raise ValueError(f'{uid}: Pass@4 requires four distinct sample lineages')
    recomputed = per_group_metrics(rows)
    for key in ('success_rate_pct','pass_at_4_pct','response_length','turns','invalid_action_fraction'):
        if metrics.get(key) is not None and not np.isclose(metrics[key],recomputed[key],atol=1e-8):
            raise ValueError(f'Saved vs recomputed {key} differ in {path}')
    by_turn = {}
    turns = sorted({t['turn'] for r in rows for t in r.get('trace',[])})
    for turn in turns:
        ts=[t for r in rows for t in r.get('trace',[]) if t['turn']==turn]
        by_turn[str(turn)] = dict(n=len(ts), invalid_fraction=mean([not t.get('action_valid',t.get('action') in (1,2,3,4)) for t in ts]),
            mean_tokens_including_eos=mean([t['response_tokens'] for t in ts if 'response_tokens' in t]),
            mean_tokens_excluding_eos=mean([t['response_tokens_excluding_eos'] for t in ts if 'response_tokens_excluding_eos' in t]))
    by_distance = {}
    for distance in sorted({str(r['task'].get('shortest_distance','unknown')) for r in rows},key=lambda s:999 if s=='unknown' else int(s)):
        selected=[r for r in rows if str(r['task'].get('shortest_distance','unknown'))==distance]
        by_distance[distance]=per_group_metrics(selected)
    summary = dict(path=str(path), raw_path=str(raw_path), split=metrics['split'],
        **recomputed, by_turn=by_turn, by_shortest_distance=by_distance,
        recorded_evaluation_seconds=metrics.get('seconds'),
        recorded_generated_tokens=metrics.get('generated_tokens'),
        raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
    return summary, dict(groups)


def training_summary(run):
    history = read_step_records(run/'train_metrics.jsonl')
    keys = ('loss','pg_loss','kl','max_token_kl','grad_norm','clipfrac','sampled_surprisal',
            'max_abs_log_ratio','rollout_is_clipped_weight','outcome_zero_variance_group_fraction',
            'learning_zero_variance_group_fraction','first_turn_invalid_fraction','invalid_action_fraction',
            'mean_response_tokens','mean_turns','resample_batches','accepted_success_rate',
            'max_parameter_change','rollout_logprob_delta_mean','rollout_logprob_delta_min',
            'rollout_logprob_delta_max','rollout_logprob_delta_p95','rollout_logprob_delta_p99')
    summaries={key:numeric_summary(finite_numbers(history,key)) for key in keys}
    incidents=[load(p) for p in (run/'recovery_history').glob('*/incident.json')]
    if any(i.get('status')!='complete' for i in incidents):
        raise ValueError(f'Incomplete recovery in {run}')
    events=read_step_records(run/'branch_events.jsonl')
    event_counts=Counter(e['kind'] for record in events for e in record.get('events',[]))
    return dict(completed_steps=len(history), last_step=history[-1]['step'] if history else 0,
        optimizer_updates=sum(r.get('optimizer_updates',0) for r in history),
        clipped_optimizer_updates=sum(r.get('clipped_optimizer_updates',0) for r in history),
        metrics=summaries,
        nonfinite_metric_counts={key:sum(isinstance(r.get(key),(int,float)) and not math.isfinite(r[key]) for r in history) for key in keys},
        probe_gradient_split=[dict(step=r['step'],probes=r['probe_gradient_split']) for r in history if 'probe_gradient_split' in r],
        final_five_step_metrics={key:numeric_summary(finite_numbers(history[-5:],key)) for key in keys},
        generated_tokens=sum(r.get('generated_tokens',0) for r in history),
        input_tokens=sum(r.get('input_tokens',0) for r in history),
        rollout_seconds=sum(r.get('rollout_seconds',0) for r in history),
        recorded_step_seconds=sum(r.get('step_seconds',0) for r in history),
        discarded_generated_tokens=sum(i.get('rolled_back_generated_tokens',0) for i in incidents),
        discarded_step_seconds=sum(i.get('rolled_back_step_seconds',0) for i in incidents),
        event_counts=dict(event_counts),
        costs_scope='Committed step logs plus known discarded iterations; model loading and incomplete failed work excluded. Endpoint/initial evaluation costs are separate.')


def task_vectors(groups):
    return {uid: np.array([np.mean([r['reward']>0 for r in rows]),any(r['reward']>0 for r in rows)],dtype=float)
            for uid,rows in groups.items()}


def compatible(a,b):
    if a.keys()!=b.keys():
        raise ValueError('Paired comparison requires exactly the same task UIDs')
    shared_seeds=0
    for uid in sorted(a):
        ta,tb=a[uid][0]['task'],b[uid][0]['task']
        if ta!=tb:
            raise ValueError(f'{uid}: map/task manifests differ; scores are not paired')
        ra,rb={r['lineage']:r for r in a[uid]},{r['lineage']:r for r in b[uid]}
        if ra.keys()!=rb.keys():
            raise ValueError(f'{uid}: sample lineages differ')
        for lineage in ra:
            xa={t['turn']:t.get('sampling_seed') for t in ra[lineage].get('trace',[])}
            xb={t['turn']:t.get('sampling_seed') for t in rb[lineage].get('trace',[])}
            for turn in xa.keys() & xb.keys():
                if xa[turn] is None or xb[turn] is None or xa[turn]!=xb[turn]:
                    raise ValueError(f'{uid}/{lineage}/turn{turn}: paired sampling seeds missing or different')
                shared_seeds+=1
    return shared_seeds


def bootstrap_differences(delta, draws, seed):
    rng=np.random.default_rng(seed)
    boot=[]
    for start in range(0,draws,1000):
        indices=rng.integers(0,len(delta),(min(1000,draws-start),len(delta)))
        boot.append(delta[indices].mean(axis=1)*100)
    boot=np.concatenate(boot)
    return {key:dict(delta_percentage_points=float(delta[:,i].mean()*100),
        percentile_95_ci=[float(v) for v in np.quantile(boot[:,i],[.025,.975])])
        for i,key in enumerate(('success','pass_at_4'))}


def compare(a,b,draws,seed):
    verified=compatible(a,b)
    av,bv=task_vectors(a),task_vectors(b)
    keys=sorted(av)
    delta=np.array([bv[k]-av[k] for k in keys])
    lost=int(sum(av[k][1]==1 and bv[k][1]==0 for k in keys))
    gained=int(sum(av[k][1]==0 and bv[k][1]==1 for k in keys))
    discordant=lost+gained
    # Exact two-sided binomial McNemar; only task-level Pass@4 is binary and independent.
    p=min(1.,2*sum(math.comb(discordant,j) for j in range(min(lost,gained)+1))/2**discordant) if discordant else 1.
    return dict(n_tasks=len(keys), direction='second minus first',
        bootstrap_unit='task, preserving the cluster of four samples', bootstrap_draws=draws,
        bootstrap_seed=seed, confidence_scope='Finite-validation-task uncertainty only; one training seed cannot estimate training variance',
        **bootstrap_differences(delta,draws,seed),
        mcnemar_pass_at_4=dict(first_only_success=int(lost),second_only_success=int(gained),
            discordant_tasks=int(discordant),exact_two_sided_p=p,
            note='Exploratory, no multiple-comparison correction; no independent-trajectory p-value reported'),
        shared_turn_seeds_verified=verified)


def parse_label(value):
    label,sep,path=value.partition('=')
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError('Expected LABEL=PATH')
    return label,Path(path)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run',action='append',type=parse_label,default=[])
    p.add_argument('--evaluation',action='append',type=parse_label,default=[])
    p.add_argument('--pair',action='append',default=[],help='BASE:VARIANT labels; endpoint and available initial comparisons')
    p.add_argument('--bootstrap-draws',type=int,default=10000)
    p.add_argument('--bootstrap-seed',type=int,default=20261014)
    p.add_argument('--output',type=Path,default=ROOT/'reports/short_validation_analysis.json')
    args=p.parse_args()
    if not args.run and not args.evaluation: p.error('Specify at least one --run or --evaluation')
    if args.bootstrap_draws<100: p.error('At least 100 bootstrap draws required')
    result=dict(scope='Validation only; prespecified final endpoints, never best checkpoint selection',
        warnings=['20 rollout steps cannot establish that degradation near step 400 is fixed.',
            'Behavioral and numerical variants are a bundled intervention, not the unpublished author implementation; do not attribute effects to a single factor.',
            'Greater RLen alone is not an improvement; inspect success, valid actions and length together.',
            'Four samples are clustered by task; confidence intervals omit training-seed variability.',
            'Test and holdout scores are not loaded or used.'],runs={},comparisons={})
    stored={}
    for label,run in args.run:
        if label in stored: raise ValueError(f'Duplicate label: {label}')
        cfg=load(run/'config.json')
        if cfg.get('run_mode')!='short_validation' or not 1<=cfg['steps']<=40:
            raise ValueError(f'Not a permitted short validation run: {run}')
        endpoint=run/f'validation_{cfg["steps"]:04d}.json'
        info=dict(path=str(run),config=cfg,status=load(run/'status.json') if (run/'status.json').exists() else {},
            planned_endpoint_step=cfg['steps'],training=training_summary(run),evaluation_cost_seconds_all_recorded=0.)
        stored[label]={}
        for phase,path in [('initial',run/'validation_0000.json'),('endpoint',endpoint)]:
            if path.exists():
                summary,groups=read_evaluation(path)
                info[phase]=summary;stored[label][phase]=groups
            else:info[phase]=dict(status='pending',expected_path=str(path))
        # Curves may be displayed; never use their maxima to choose the endpoint.
        info['validation_curve']=[]
        for path in sorted(run.glob('validation_*.json')):
            m=load(path)
            if m.get('split') not in ALLOWED_SPLITS: raise ValueError('Non-validation curve file')
            info['validation_curve'].append(dict(step=int(path.stem.split('_')[-1]),
                **{k:m.get(k) for k in ('success_rate_pct','pass_at_4_pct','response_length','first_turn_invalid_fraction')}))
            info['evaluation_cost_seconds_all_recorded']+=m.get('seconds',0)
        # train.py historically calls the final validation artifact final_test.
        # Inspect only its split label here; never read a raw test/holdout file.
        final=run/'final_validation.json'
        if final.exists():
            m=load(final)
            if m.get('split') not in ALLOWED_SPLITS: raise ValueError('Unexpected final split')
            info['evaluation_cost_seconds_all_recorded']+=m.get('seconds',0)
        if 'initial' in stored[label] and 'endpoint' in stored[label]:
            info['endpoint_minus_initial']=compare(stored[label]['initial'],stored[label]['endpoint'],args.bootstrap_draws,args.bootstrap_seed)
        result['runs'][label]=info
    for label,path in args.evaluation:
        if label in stored:raise ValueError(f'Duplicate label: {label}')
        summary,groups=read_evaluation(path)
        stored[label]={'endpoint':groups}
        result['runs'][label]=dict(kind='explicit_frozen_weight_validation',endpoint=summary)
    for pair in args.pair:
        a,sep,b=pair.partition(':')
        if not sep or a not in stored or b not in stored:raise ValueError(f'Unknown pair: {pair}')
        comparisons={}
        for phase in ('initial','endpoint'):
            if phase in stored[a] and phase in stored[b]:
                comparisons[phase]=compare(stored[a][phase],stored[b][phase],args.bootstrap_draws,args.bootstrap_seed)
            else:comparisons[phase]=dict(status='pending_or_unavailable')
        if all(phase in stored[label] for label in (a,b) for phase in ('initial','endpoint')):
            ai,ae,bi,be=[task_vectors(stored[label][phase]) for label,phase in ((a,'initial'),(a,'endpoint'),(b,'initial'),(b,'endpoint'))]
            keys=sorted(ai)
            delta=np.array([(be[k]-bi[k])-(ae[k]-ai[k]) for k in keys])
            comparisons['difference_in_training_changes']=dict(direction='(variant endpoint-initial) minus (baseline endpoint-initial)',
                **bootstrap_differences(delta,args.bootstrap_draws,args.bootstrap_seed),
                note='Exploratory descriptive contrast; prompt, reward, normalization, routing and numerical variants remain bundled')
        result['comparisons'][pair]=comparisons
    atomic_json(args.output,result)
    print(json.dumps(dict(output=str(args.output),runs=list(result['runs']),comparisons=list(result['comparisons'])),ensure_ascii=False))


if __name__=='__main__':
    main()
