"""Training-task UID-clustered gate; reference PaTR400 excluded from decisions."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import patr_core as core
B1=Path('/home/vipuser/golden-next-20261031/code/runs/b1')

def decision(u,r,lower,total_tokens,baselines):
    return dict(uct_strictly_more=u>r,uid_cluster_ci_lower_positive=lower>0,
      not_below_best_b1_k2=u>=max(baselines['K2_patr']['mixed'],baselines['K2_tree_random']['mixed']),
      token_within_115pct_both_b1_k2=all(total_tokens<=1.15*baselines[f'K2_{m}']['tokens'] for m in ('patr','tree_random')))

def main():
    records={};models={};baselines={};total_tokens=0
    for model in ('sft','sft_grpo400','sft_patr400'):
        out=ROOT/'runs/g1'/model
        assert json.loads((out/'status.json').read_text())['status']=='complete'
        assert json.loads((out/'gpu_consistency.json').read_text())['passed']
        rows={};costs={}
        for batch in range(1,4):
            data=json.loads((out/f'batch{batch}.json').read_text())
            assert not data['replay_skips'],'Replay skips make G1 incomplete; preserve and investigate'
            for arm,result in data['arms'].items():
                rows.setdefault(arm,[]).extend(result['rows']);costs[arm]=costs.get(arm,0)+result['cost']['generated_tokens']
            if model=='sft_patr400':continue
            u=data['arms']['F2_uct']['rows'];r=data['arms']['F2_uniform']['rows'];assert [x['uid'] for x in u]==[x['uid'] for x in r]
            for x,y in zip(u,r):records.setdefault(x['uid'],[]).append(int(x['mixed'])-int(y['mixed']))
            for k in (2,5):
                for method in ('patr','tree_random'):
                    old=json.loads((B1/model/f'k{k}_{method}_batch{batch}.json').read_text())
                    assert [x['task']['uid'] for x in old['groups']]==[x['uid'] for x in u]
                    label=f'K{k}_{method}';b=baselines.setdefault(label,dict(mixed=0,tokens=0))
                    b['mixed']+=old['mixed_groups'];b['tokens']+=old['actual_cost']['generated_tokens']
        models[model]={}
        for arm,xs in rows.items():
            subsets={key:[x for x in xs if x[key]] for key in ('initial_all_success','initial_all_failure')}
            models[model][arm]=dict(mixed=sum(x['mixed'] for x in xs),tasks=len(xs),generated_tokens=costs[arm],
              rescue={k:dict(n=len(v),rescued=sum(x['rescued'] for x in v),rate=sum(x['rescued'] for x in v)/len(v) if v else None) for k,v in subsets.items()},
              new_success_rate=sum(x['new_successes'] for x in xs)/sum(x['new_leaves'] for x in xs),
              dup_first_action_rate=sum(x['duplicate_first'] for x in xs)/sum(x['duplicate_denominator'] for x in xs) if sum(x['duplicate_denominator'] for x in xs) else None)
        if model!='sft_patr400':total_tokens+=costs['F2_uct']
    values=[records[k] for k in sorted(records)];sums=np.array([sum(v) for v in values]);counts=np.array([len(v) for v in values]);rng=np.random.default_rng(20261101)
    samples=[]
    for _ in range(20000):
        idx=rng.integers(0,len(values),len(values));samples.append(float(sums[idx].sum()/counts[idx].sum()))
    ci=np.quantile(samples,[.025,.975]).tolist()
    u=sum(models[m]['F2_uct']['mixed'] for m in ('sft','sft_grpo400'));r=sum(models[m]['F2_uniform']['mixed'] for m in ('sft','sft_grpo400'))
    checks=decision(u,r,ci[0],total_tokens,baselines)
    core.atomic_json(ROOT/'runs/g1/gate.json',dict(pass_gate=all(checks.values()),checks=checks,uct_mixed=u,uniform_mixed=r,uid_clusters=len(values),mixed_difference_ci95=ci,
        generated_tokens=total_tokens,baseline=baselines,models=models,budget_for_G2=2,reference_model_excluded=True,
        limitations=['one training-task screen; no validation/test selection','F1 and three other selectors are descriptive only','token cap conservatively applied to each B1 K2 comparator; shared computation not amortized']))
    print(json.dumps(dict(pass_gate=all(checks.values()),checks=checks,ci=ci)))
if __name__=='__main__':main()
