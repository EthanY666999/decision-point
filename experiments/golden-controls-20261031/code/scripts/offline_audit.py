"""Read-only audit; no training decisions from test outcomes."""
import collections, gzip, hashlib, json, sys, statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from patr_core import make_env, make_datasets, atomic_json, distance, FrozenLakeAgent
OLD=Path('/home/vipuser/golden-sft-patr-20261031/code')
EXT=Path('/home/vipuser/golden-patr-extended800-20261031/code')
OUT=ROOT/'runs/audit';OUT.mkdir(parents=True,exist_ok=True)
CFG={'max_turns':10,'is_slippery':False}
def rows(files):
    return [json.loads(s) for p in files for s in p.open()]
def geom(t):
    e=make_env(t,CFG)
    return tuple(''.join(c.decode() for c in row) for row in e.desc)
def summary(rs):
    groups=collections.defaultdict(list)
    for r in rs:groups[r['task']['uid']].append(r)
    assert all(len(g)==4 for g in groups.values())
    legal=[r for r in rs if r['trace'] and all(t['action'] in (1,2,3,4) for t in r['trace'])]
    actions=[t['action'] for r in rs for t in r['trace']]
    return dict(tasks=len(groups),trajectories=len(rs),success_pct=100*sum(r['reward']>0 for r in rs)/len(rs),pass4_pct=100*sum(any(r['reward']>0 for r in g) for g in groups.values())/len(groups),invalid_pct=100*sum(a not in (1,2,3,4) for a in actions)/len(actions),zero_invalid_n=len(legal),zero_invalid_success_pct=100*sum(r['reward']>0 for r in legal)/len(legal) if legal else None)
def audit_overlap(data):
    training=set(map(geom,data['train']))
    tasks=json.loads((OLD.parent/'results/supplemental500/tasks.json').read_text())
    overlap={t['uid'] for t in tasks if geom(t) in training}
    assert len(overlap)==91, f'Expected audit91; observed{len(overlap)}'
    paths={
      'old_patr':[OLD.parent/f'results/supplemental500/patr/batch{i}.jsonl' for i in range(1,6)],
      'sft':[OLD/f'runs/golden_sft/supplemental500/sft/batch{i}.jsonl' for i in range(1,6)],
      'sft_grpo400':[OLD/f'runs/golden_sft/supplemental500/grpo/batch{i}.jsonl' for i in range(1,6)],
      'sft_patr400':[OLD/f'runs/golden_sft/supplemental500/patr/batch{i}.jsonl' for i in range(1,6)],
      'sft_patr500':[EXT/f'runs/finish500/supplemental500/batch{i}.jsonl' for i in range(1,6)]}
    result={'definition':'Exact labelled map including S/G/H; no rotation/reflection equivalence. Compared against all 10000 RL training tasks, not only sampled tasks. Post-hoc sensitivity analysis; original500 remains unchanged. Does not certify absence from pretrained/SFT data.', 'excluded_uids':sorted(overlap),'excluded_n':len(overlap),'retained_n':500-len(overlap),'models':{}}
    per_model={}
    for name,files in paths.items():
        rs=rows(files);assert len(rs)==2000
        filtered=[r for r in rs if r['task']['uid'] not in overlap]
        result['models'][name]={'original500':summary(rs),'unseen409':summary(filtered),'seen91':summary([r for r in rs if r['task']['uid'] in overlap])}
        per_model[name]={u:sum(r['reward']>0 for r in filtered if r['task']['uid']==u)/4 for u in {r['task']['uid'] for r in filtered}}
    # Paired map-cluster bootstrap accounts for duplicated geometry among test tasks.
    uid_geom={t['uid']:geom(t) for t in tasks if t['uid'] not in overlap};clusters=collections.defaultdict(list)
    for u,g in uid_geom.items():clusters[g].append(u)
    rng=np.random.default_rng(20261031);cluster_values=list(clusters.values());comparisons={}
    for name in ['sft_grpo400','sft_patr400','sft_patr500']:
        diffs={u:per_model[name][u]-per_model['old_patr'][u] for u in uid_geom}
        nums=np.array([sum(diffs[u] for u in us) for us in cluster_values]);dens=np.array([len(us) for us in cluster_values])
        draws=rng.integers(len(nums),size=(10000,len(nums)));boot=nums[draws].sum(1)/dens[draws].sum(1)*100
        comparisons[name+'_minus_old_patr']={'success_delta_pp':100*sum(diffs.values())/len(diffs),'paired_geometry_cluster_CI95':np.quantile(boot,[.025,.975]).tolist()}
    result['unique_unseen_geometries']=len(clusters);result['comparisons']=comparisons
    result['caveat']='Conditional zero-invalid success compares selected, potentially different trajectories. N/A if denominator zero. Bootstrap covers sampled maps only, not training-seed uncertainty.'
    atomic_json(OUT/'overlap_report.json',result)
    # Freeze shared state manifest before inference. Equal map round-robin sampling, unique (task,state).
    candidates=[];rng=np.random.default_rng(20261031)
    for task in data['test']:
        env=make_env(task,CFG);start=int(env.s);q=[start];seen={start}
        for pos in q:
            r,c=divmod(pos,env.ncol)
            for dr,dc in [(0,-1),(1,0),(0,1),(-1,0)]:
                rr,cc=r+dr,c+dc;p=rr*env.ncol+cc
                if 0<=rr<env.nrow and 0<=cc<env.ncol and p not in seen and env.desc[rr,cc]!=b'H':seen.add(p);q.append(p)
        states=[]
        for pos in sorted(seen):
            env.s=pos;d=distance(env)
            if d is None or not 1<=d<=10:continue
            optimal=[];r,c=divmod(pos,env.ncol)
            for a,(dr,dc) in enumerate([(0,-1),(1,0),(0,1),(-1,0)],1):
                rr,cc=r+dr,c+dc
                if not (0<=rr<env.nrow and 0<=cc<env.ncol) or env.desc[rr,cc]==b'H':continue
                env.s=rr*env.ncol+cc
                if distance(env)==d-1:optimal.append(a)
            assert optimal;env.s=pos
            agent=FrozenLakeAgent(max_steps=10);agent.update_from_env(env.render(),0.,False,{})
            states.append(dict(task=task,state=pos,shortest_distance=d,optimal_actions=optimal,messages=agent.chat_completions,train_geometry_seen=geom(task) in training))
        rng.shuffle(states);candidates.append(states)
    selected=[]
    while len(selected)<1000 and any(candidates):
        for ss in candidates:
            if ss and len(selected)<1000:selected.append(ss.pop())
    for i,s in enumerate(selected):s['id']=i
    manifest={'protocol':'Fixed test100 maps; unique reachable safe states with shortest goal distance1..10; round-robin maps, RNG20261031; fresh standard prompt, no prior history, ten remaining steps; greedy decoding; frozen before inference; diagnostic only, not training selection.', 'states':selected,'n_states':len(selected),'n_tasks':len({s['task']['uid'] for s in selected})}
    atomic_json(OUT/'state_manifest.json',manifest)
    atomic_json(OUT/'state_manifest_identity.json',{'sha256':hashlib.sha256((OUT/'state_manifest.json').read_bytes()).hexdigest(),'n_states':len(selected)})
    print('OVERLAP_AND_MANIFEST_COMPLETE',len(selected),flush=True)
def audit_branches(data,root,label):
    run=root/'runs/formal/patr_seed42';events=rows([run/'branch_events.jsonl']);records=[];paired=[];stats=collections.Counter();perstep=[]
    for er in events:
        step=er['step'];ev=er['events'];forks=[e for e in ev if e['kind']=='fork_candidates'];stats['steps']+=1;stats['fork_events']+=len(forks);perstep.append(len(forks));stats['prunes']+=sum(e['kind']=='prune' for e in ev)
        # Fixed training task draw, independent of rollout policy.
        trng=np.random.default_rng(np.random.SeedSequence([42,step,1701]));tasks=[data['train'][int(i)] for i in trng.choice(len(data['train']),64,replace=False)]
        grouped=collections.defaultdict(list)
        with gzip.open(run/f'rollouts_{step:04d}.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line);grouped[r['task']['uid']].append({'lineage':r['lineage'],'success':int(r['reward']>0)})
        assert set(grouped)=={t['uid'] for t in tasks}
        for e in forks:
            cs=e['candidates'];scores=[c['score'] for c in cs];selected=[c for c in cs if c['selected']];uid=tasks[e['task_id']]['uid']
            stats['singleton']+=len(cs)==1;stats['all_scores_tied']+=len(set(scores))==1
            stats['multiple_candidates']+=len(cs)>=2;stats['tied_multiple']+=len(cs)>=2 and len(set(scores))==1
            stats['selection_opportunities']+=len(cs)>2 and len(set(scores))>1
            prs=[]
            for c in selected:
                descendants=[r for r in grouped[uid] if r['lineage'].startswith(c['lineage']+'.')]
                assert len(descendants)==2,(step,uid,c,len(descendants))
                ys=[r['success'] for r in descendants];mixed=int(min(ys)<max(ys))
                row=dict(step=step,uid=uid,task_id=e['task_id'],parent=c['lineage'],score=c['score'],child_success_mean=statistics.mean(ys),child_mixed=mixed)
                records.append(row);prs.append(row)
            if len(prs)==2 and prs[0]['score']!=prs[1]['score']:
                low,high=sorted(prs,key=lambda x:x['score']);paired.append(dict(step=step,uid=uid,high_score=high['score'],low_score=low['score'],mixed_delta=high['child_mixed']-low['child_mixed'],success_delta=high['child_success_mean']-low['child_success_mean']))
        if step%50==0:print(label,step,flush=True)
    stats['different_selected_pair_events']=len(paired);stats['total_task_groups']=stats['steps']*64
    def boot(key):
        clusters=collections.defaultdict(list)
        for p in paired:clusters[p['uid']].append(p[key])
        cs=list(clusters.values());num=np.array([sum(v) for v in cs]);den=np.array([len(v) for v in cs]);rng=np.random.default_rng(42)
        if not cs:return None
        inds=rng.integers(len(cs),size=(10000,len(cs)));vals=num[inds].sum(1)/den[inds].sum(1)
        return dict(mean=float(num.sum()/den.sum()),paired_uid_cluster_CI95=np.quantile(vals,[.025,.975]).tolist(),clusters=len(cs))
    result={'counts':dict(stats),'forked_tasks_per_step_mean':statistics.mean(perstep),'last50_forked_tasks_mean':statistics.mean(perstep[-50:]),'three_plus_one_eligible_task_pct':100*len(paired)/stats['total_task_groups'],'paired_high_minus_low_mixed':boot('mixed_delta'),'paired_high_minus_low_child_success':boot('success_delta'),'parent_score_bins':{},'limitations':'Observational selected-parent data only; no outcomes for unselected counterfactual continuations. Two children do not identify effect of adding a third. Higher success probability need not imply higher mixed probability. UID bootstrap does not model training nonstationarity, geometry duplicates or seed uncertainty. Singleton events are reported separately from genuine ties.'}
    for lo,hi in [(0,.25),(.25,.5),(.5,.75),(.75,1.01)]:
        rr=[r for r in records if lo<=r['score']<hi];result['parent_score_bins'][str((lo,hi))]={'n':len(rr),'child_mixed_fraction':statistics.mean(r['child_mixed'] for r in rr) if rr else None,'child_success_fraction':statistics.mean(r['child_success_mean'] for r in rr) if rr else None}
    atomic_json(OUT/f'branch_report_{label}.json',result)
    (OUT/f'parent_outcomes_{label}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (OUT/f'paired_parents_{label}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in paired))
if __name__=='__main__':
    data=make_datasets();audit_overlap(data)
    audit_branches(data,OLD,'patr400');audit_branches(data,EXT,'patr401_500')
    print('ALL_OFFLINE_COMPLETE',flush=True)
