"""CPU-only, read-only comparisons of completed controls; never selects training settings."""
import collections
import functools
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patr_core import distance, make_env

OLD = Path('/home/vipuser/golden-sft-patr-20261031/code')
OUT = ROOT / 'runs/audit'

def read(paths):
    return [json.loads(line) for p in paths for line in p.read_text().splitlines()]

@functools.lru_cache(None)
def map_info(task_json):
    env = make_env(json.loads(task_json), dict(is_slippery=False, max_turns=10))
    return tuple(''.join(c.decode() for c in row) for row in env.desc), distance(env)

def summarize(rs):
    groups = collections.defaultdict(list)
    for r in rs:
        groups[r['task']['uid']].append(r)
    assert all(len(v) == 4 for v in groups.values())
    legal = [r for r in rs if r['trace'] and all(t['action'] in (1, 2, 3, 4) for t in r['trace'])]
    actions = [t['action'] for r in rs for t in r['trace']]
    successes = [r for r in rs if r['reward'] > 0]
    excess = [len(r['trace']) - map_info(json.dumps(r['task'], sort_keys=True))[1] for r in successes]
    invalid = [sum(t['action'] not in (1, 2, 3, 4) for t in r['trace']) for r in successes]
    assert all(x >= 0 for x in excess)
    return dict(tasks=len(groups), trajectories=len(rs),
        success_pct=100*len(successes)/len(rs),
        pass4_pct=100*sum(any(r['reward'] > 0 for r in g) for g in groups.values())/len(groups),
        invalid_pct=100*sum(a not in (1,2,3,4) for a in actions)/len(actions),
        first_turn_legal_pct=100*sum(bool(r['trace']) and r['trace'][0]['action'] in (1,2,3,4) for r in rs)/len(rs),
        zero_invalid_n=len(legal),
        zero_invalid_success_pct=100*sum(r['reward'] > 0 for r in legal)/len(legal) if legal else None,
        successful_mean_turns=statistics.mean(len(r['trace']) for r in successes),
        successful_excess_steps=statistics.mean(excess),
        successful_excess_excluding_invalid=statistics.mean(e-i for e,i in zip(excess,invalid))), groups

def paired(new, old):
    assert set(new) == set(old)
    clusters = collections.defaultdict(list)
    for uid in sorted(new):
        assert new[uid][0]['task'] == old[uid][0]['task']
        def score(g):
            return np.array([sum(r['reward'] > 0 for r in g)/4, float(any(r['reward'] > 0 for r in g))])
        geometry = map_info(json.dumps(new[uid][0]['task'], sort_keys=True))[0]
        clusters[geometry].append(score(new[uid])-score(old[uid]))
    nums = np.array([np.sum(v, axis=0) for v in clusters.values()])
    dens = np.array([len(v) for v in clusters.values()])
    rng = np.random.default_rng(20261031)
    draws = rng.integers(len(nums), size=(10000,len(nums)))
    boot = nums[draws].sum(1)/dens[draws].sum(1)[:,None]*100
    return dict(delta_success_pp=float(nums.sum(0)[0]/dens.sum()*100),
        delta_pass4_pp=float(nums.sum(0)[1]/dens.sum()*100),
        paired_geometry_cluster_CI95=np.quantile(boot,[.025,.975],axis=0).T.tolist(),
        geometry_clusters=len(nums), replicates=10000)

def train_stats(path):
    rs = read([path/'train_metrics.jsonl'])
    assert [r['step'] for r in rs] == list(range(1,401))
    assert all(math.isfinite(r[k]) for r in rs for k in ['loss','grad_norm','max_parameter_change'])
    assert all(r['max_parameter_change'] > 0 for r in rs)
    mixed = []
    fork_counts = []
    for row in read([path/'branch_events.jsonl']):
        fork_counts.append(sum(e['kind']=='fork_candidates' for e in row['events']))
    for step in range(1,401):
        groups = collections.defaultdict(list)
        with gzip.open(path/f'rollouts_{step:04}.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line); groups[r['task']['uid']].append(r['reward'])
        assert len(groups)==64
        mixed.append(sum(len(set(g)) > 1 for g in groups.values())/64)
    return dict(steps=400, all_finite=True, every_step_updated=True,
        generated_tokens=sum(r['generated_tokens'] for r in rs),
        executed_actions=sum(r['executed_actions'] for r in rs),
        step_seconds=sum(r['step_seconds'] for r in rs),
        elapsed_seconds=rs[-1]['elapsed_seconds'],
        mixed_retained_groups_pct=100*statistics.mean(mixed),
        last40_mixed_retained_groups_pct=100*statistics.mean(mixed[-40:]),
        fork_events=sum(fork_counts), fork_events_per_task_pct=100*sum(fork_counts)/(400*64),
        mean_retained_trajectories_per_task=statistics.mean(r['episodes']/64 for r in rs),
        hardware='RTX 4090 48 GiB' if path.is_relative_to(ROOT) else 'A100',
        note='Wall time differs by GPU. Retained branches include shared prefixes. Costs exclude SFT; step time excludes validation/export overhead.')

def main():
    excluded=set(json.loads((OUT/'overlap_report.json').read_text())['excluded_uids'])
    assert len(excluded)==91
    models = {'old_patr':None,'sft_patr400':OLD/'runs/formal/patr_seed42',
              'sft_grpo400':OLD/'runs/formal/grpo_seed42',
              'sft_tree_random400':ROOT/'runs/formal/tree_random_seed42'}
    if (ROOT/'runs/audit/supplemental500/grpo/summary.json').exists():
        models['sft_grpo4_400']=ROOT/'runs/formal/grpo_seed42'
    result={'protocol':'Fixed main100, original supplemental500, and post-hoc filtered409 reported separately. No test-based selection.',
        'limitations':['Single training seed; geometry bootstrap does not cover training-seed uncertainty.',
        'TreeRandom randomizes selection AND disables score pruning; not a pure selection ablation.',
        'GRPO4 is a group-size control, not exactly compute-matched.',
        'Zero-invalid and successful-path metrics condition on different trajectory subsets; not causal decision-quality estimates.',
        'Filtered409 excludes RL training map geometry only, not necessarily SFT/pretraining overlap.'], 'evaluations':{},'training':{}}
    for split in ['main100','supplemental500','filtered409']:
        summaries={};groups={}
        for name,path in models.items():
            if split=='main100':
                files=[OLD.parent/'results/main100/runs/patr_seed42/final_test.jsonl'] if path is None else [path/'final_test.jsonl']
            else:
                if path is None: base=OLD.parent/'results/supplemental500/patr'
                elif name=='sft_patr400': base=OLD/'runs/golden_sft/supplemental500/patr'
                elif name=='sft_grpo400': base=OLD/'runs/golden_sft/supplemental500/grpo'
                else: base=ROOT/'runs/audit/supplemental500'/('tree_random' if name=='sft_tree_random400' else 'grpo')
                files=[base/f'batch{i}.jsonl' for i in range(1,6)]
            rs=read(files)
            if split=='filtered409': rs=[r for r in rs if r['task']['uid'] not in excluded]
            summaries[name],groups[name]=summarize(rs)
            assert summaries[name]['tasks']=={'main100':100,'supplemental500':500,'filtered409':409}[split]
        comparisons={}
        for name in ['sft_tree_random400','sft_grpo4_400']:
            if name in groups:
                for ref in ['old_patr','sft_patr400','sft_grpo400']:
                    comparisons[name+'_minus_'+ref]=paired(groups[name],groups[ref])
        result['evaluations'][split]={'models':summaries,'comparisons':comparisons,'task_dictionaries_verified':True}
    for name,path in models.items():
        if path is not None: result['training'][name]=train_stats(path)
    dest=OUT/('completed_controls_comparison.json' if 'sft_grpo4_400' in models else 'tree_random_completed_comparison.json')
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'report':str(dest),'evaluations':result['evaluations'],'training':result['training']},indent=2))

if __name__=='__main__': main()
