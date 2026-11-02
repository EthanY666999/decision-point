"""Predeclared B1 gate, based exclusively on fixed training-task samples."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def interval(values):
    # Cluster all repeated model/batch observations of the same task UID together.
    clusters=defaultdict(list)
    for uid,value in values:clusters[uid].append(value)
    if len(clusters)<2:return {'mean':None,'CI95':None,'clusters':len(clusters)}
    nums=np.array([sum(x) for x in clusters.values()]);dens=np.array([len(x) for x in clusters.values()])
    rng=np.random.default_rng(20261031)
    ids=rng.integers(len(nums),size=(10000,len(nums)))
    draws=nums[ids].sum(1)/dens[ids].sum(1)
    return {'mean':float(nums.sum()/dens.sum()),'CI95':np.quantile(draws,[.025,.975]).tolist(),
            'clusters':len(clusters),'observations':int(dens.sum())}

def main():
    rows=[json.loads(p.read_text()) for p in sorted((ROOT/'runs/b1').glob('*/k*_batch*.json'))]
    assert len(rows)==48, f'Need 48 complete cells, have {len(rows)}'
    by={(r['model'],r['K'],r['method'],r['batch']):r for r in rows}
    result={'training_tasks_only':True,'K_results':{},'pass':False,'selected_K':None,
        'gate':'Pooled across both predeclared models: PaTR mixed groups > Random, and PaTR selected-minus-unselected shadow success lower 95% CI > 0. Choose largest mixed-group gain; smaller K breaks ties.',
        'limitations':['Pointwise 95% intervals across four candidate K values; exploratory gate, not a confirmatory significance claim.',
        'Shadow estimand is two fresh continuations per parent at the first comparable decision per task, without later branching. It is not the realized full-tree descendant outcome.',
        'PaTR retains score pruning; Random disables it. Report actual pruning counts; this is not a pure selection-only ablation.',
        'Fixed leaf ceilings do not equal fixed token or interaction costs.']}
    for k in [1,2,3,5]:
        mixed=[];shadow_by_method=defaultdict(list);details={}
        for model in ['sft','sft_grpo400']:
            details[model]={}
            for method in ['patr','tree_random']:
                cells=[by[model,k,method,b] for b in [1,2,3]]
                vals=[]
                for r in cells:
                    groups=defaultdict(lambda:defaultdict(list))
                    for s in r['shadow']:groups[s['task_uid']][bool(s['selected'])].append(float(s['reward']>0))
                    for uid,g in groups.items():
                        assert True in g and False in g
                        vals.append((uid,float(np.mean(g[True])-np.mean(g[False]))))
                shadow_by_method[method].extend(vals)
                details[model][method]={f:sum(r[f] for r in cells) for f in ['mixed_groups','forked_tasks','fork_points','tied_fork_points','multi_parent_fork_points','tied_multi_parent_forks','score_prunes']}
                details[model][method].update(task_groups=192,shadow_success_delta=interval(vals),
                    actual_cost={f:sum(r['actual_cost'][f] for r in cells) for f in ['generated_tokens','executed_actions','seconds']},
                    shadow_cost={f:sum(r['shadow_cost'][f] for r in cells) for f in ['generated_tokens','executed_actions','seconds']})
            for b in [1,2,3]:
                p={g['task']['uid']:g for g in by[model,k,'patr',b]['groups']}
                r={g['task']['uid']:g for g in by[model,k,'tree_random',b]['groups']}
                assert p.keys()==r.keys()
                mixed.extend((uid,float(p[uid]['mixed'])-float(r[uid]['mixed'])) for uid in p)
        delta=interval(shadow_by_method['patr']);gain=sum(x for _,x in mixed)
        passed=gain>0 and delta['CI95'] is not None and delta['CI95'][0]>0
        result['K_results'][str(k)]=dict(details=details,mixed_group_gain=int(gain),mixed_fraction_delta=interval(mixed),
            patr_shadow_delta=delta,random_shadow_delta=interval(shadow_by_method['tree_random']),passes=passed)
    candidates=[int(k) for k,v in result['K_results'].items() if v['passes']]
    if candidates:
        result['pass']=True
        result['selected_K']=max(candidates,key=lambda k:(result['K_results'][str(k)]['mixed_group_gain'],-k))
    out=ROOT/'runs/b1/gate.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'gate':str(out),'pass':result['pass'],'selected_K':result['selected_K']}))

if __name__=='__main__':main()
