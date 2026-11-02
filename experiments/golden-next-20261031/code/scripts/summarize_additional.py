"""A3--A6: reuse completed raw evidence; no GPU and no test-based choices."""
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
C=Path('/home/vipuser/golden-controls-20261031/code')
O=Path('/home/vipuser/golden-sft-patr-20261031/code')
E=Path('/home/vipuser/golden-patr-extended800-20261031/code')
sys.path.insert(0,str(C/'scripts'));sys.path.insert(0,str(C))
from compare_completed_controls import read,summarize,paired

def uid_paired(a,b):
    assert a.keys()==b.keys()
    def score(rows):return [sum(r['reward']>0 for r in rows)/4,any(r['reward']>0 for r in rows)]
    ds=np.array([np.array(score(a[u]),dtype=float)-score(b[u]) for u in sorted(a)])
    rng=np.random.default_rng(20261031);ix=rng.integers(len(ds),size=(10000,len(ds)))
    return dict(delta_pp=(ds.mean(0)*100).tolist(),CI95_pp=np.quantile(ds[ix].mean(1)*100,[.025,.975],axis=0).T.tolist(),unit='task UID, preserving four trajectories',replicates=10000)

def main():
    mainpaths={
      'old_grpo':O.parent/'results/main100/v1_baselines/grpo_seed42/final_test.jsonl',
      'old_patr':O.parent/'results/main100/runs/patr_seed42/final_test.jsonl',
      'old_tree_random':O.parent/'results/main100/runs/tree_random_seed42/final_test.jsonl',
      'sft':O/'runs/golden_sft/sft_test.jsonl',
      'sft_grpo400':O/'runs/formal/grpo_seed42/final_test.jsonl',
      'sft_patr400':O/'runs/formal/patr_seed42/final_test.jsonl',
      'sft_patr500':E/'runs/formal/patr_seed42/final_test.jsonl',
      'sft_tree_random400':C/'runs/formal/tree_random_seed42/final_test.jsonl'}
    supp={f'old_{m}':O.parent/f'results/supplemental500/{m}' for m in ['grpo','patr','tree_random']}
    supp.update({k:O/f'runs/golden_sft/supplemental500/{m}' for k,m in [('sft','sft'),('sft_grpo400','grpo'),('sft_patr400','patr')]})
    supp.update(sft_patr500=E/'runs/finish500/supplemental500',sft_tree_random400=C/'runs/audit/supplemental500/tree_random')
    if (C/'runs/audit/supplemental500/grpo/summary.json').exists():
        mainpaths['sft_grpo4_400']=C/'runs/formal/grpo_seed42/final_test.jsonl'
        supp['sft_grpo4_400']=C/'runs/audit/supplemental500/grpo'
    excluded=set(json.loads((C/'runs/audit/overlap_report.json').read_text())['excluded_uids']);assert len(excluded)==91
    result={'evaluations':{},'A3':'PaTR500 supplemental500 already complete; raw evidence reused.',
        'A6':'3+1 applies to 631/25600=2.46% task groups. High-minus-low parent mixed-child outcome difference +2.06 pp, UID-cluster 95% CI [-0.79,4.96]; no demonstrated contrast benefit, no 3+1 training. The often-quoted 72% ties includes singleton decisions; multi-parent ties are 2038/3192=63.85%.',
        'limitations':['Filtered409 is post-hoc RL-map overlap sensitivity, not a replacement test or a guarantee of SFT/pretraining disjointness.',
        'Paired test intervals do not include training-seed uncertainty; baseline seed42 remains a single training run.',
        'TreeRandom also disables score pruning. Costs must be reported separately.',
        'Conditional zero-invalid success and successful-path length use different selected trajectory sets.']}
    for split in ['main100','supplemental500','filtered409']:
        models={};groups={}
        for label,mp in mainpaths.items():
            paths=[mp] if split=='main100' else [supp[label]/f'batch{i}.jsonl' for i in range(1,6)]
            rs=read(paths)
            if split=='filtered409':rs=[r for r in rs if r['task']['uid'] not in excluded]
            models[label],groups[label]=summarize(rs)
            assert models[label]['tasks']=={'main100':100,'supplemental500':500,'filtered409':409}[split]
        comparisons={}
        for a,b in itertools.combinations(['sft_grpo400','sft_patr400','sft_tree_random400'],2):
            comparisons[a+'_minus_'+b]={'task_paired':uid_paired(groups[a],groups[b]),'geometry_cluster_sensitivity':paired(groups[a],groups[b])}
        result['evaluations'][split]=dict(models=models,comparisons=comparisons)
    cross=ROOT/'runs/evaluations/cross_gpu_main100/sft_patr400/metrics.jsonl'
    if cross.exists():
        fresh,fg=summarize(read([cross]));original,og=summarize(read([mainpaths['sft_patr400']]))
        result['A2']=dict(original_A100=original,fresh_4090=fresh,paired=uid_paired(fg,og),
                         note='Observed rerun difference includes device, fresh process and numerical/scheduling effects; cannot isolate hardware causality.')
    result['A1']={p.parent.name:json.loads(p.read_text()) for p in (ROOT/'runs/evaluations/map8').glob('*/summary.json')}
    out=ROOT/'runs/analysis';out.mkdir(parents=True,exist_ok=True)
    (out/'additional_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'report':str(out/'additional_comparison.json'),'models':list(mainpaths),'map8_completed':list(result['A1'])}))

if __name__=='__main__':main()
