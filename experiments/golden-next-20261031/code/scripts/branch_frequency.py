"""B1: paired training-task sampling plus separately charged shadow continuations.

No optimizer, no weight writes, no validation/test selection. The core is imported
from a frozen source; only its routing callable is replaced within this process.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from branch_budget import BudgetRouter

ROOT=Path(__file__).resolve().parents[1]
FROZEN=Path('/home/vipuser/golden-controls-20261031/code')
OLD=Path('/home/vipuser/golden-sft-patr-20261031/code')
sys.path.insert(0,str(FROZEN))
import patr_core as core

def shadow_rollouts(router,tasks,engine,tokenizer,cfg,batch):
    from vllm import SamplingParams
    active=[];meta={};finished=[];tokens=actions=inputs=0
    for ti,parents in sorted(router.snapshots.items()):
        for parent,selected,score in parents:
            for rep in range(2):
                child=parent.clone(f'.shadow{rep}')
                meta[id(child)]=dict(task_uid=tasks[ti]['uid'],parent_lineage=parent.lineage,
                    parent_turn=parent.turns,selected=selected,score=score,replicate=rep)
                active.append(child)
    start=time.monotonic()
    while active:
        ready=[];params=[]
        for b in active:
            rem=min(cfg['max_response_tokens']-b.ntokens,cfg['max_context_tokens']-len(b.ids))
            if rem<=0:b.terminal='budget_limit';finished.append(b);continue
            m=meta[id(b)]
            key=[20261031,batch,m['task_uid'],m['parent_lineage'],m['parent_turn'],m['replicate'],b.turns]
            seed=int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:8],'big')%(2**31-1)
            params.append(SamplingParams(temperature=cfg['temperature'],top_p=1.,top_k=-1,
                max_tokens=rem,logprobs=1,seed=seed,stop_token_ids=[tokenizer.eos_token_id]))
            ready.append(b);inputs+=len(b.ids)
        if not ready:break
        outputs=engine.generate([{'prompt_token_ids':b.ids} for b in ready],params,use_tqdm=False)
        active=[]
        for b,result in zip(ready,outputs):
            n=len(b.actions);tokens+=len(result.outputs[0].token_ids)
            core.apply_response(b,result.outputs[0],tokenizer,cfg);actions+=len(b.actions)-n
            if not b.terminal and len(b.actions)>=cfg['repeat_limit'] and len(set(b.actions[-cfg['repeat_limit']:]))==1:
                b.terminal='pruned_repeat'
            if b.terminal:finished.append(b)
            else:active.append(b)
    rows=[dict(**meta[id(b)],reward=b.reward,terminal=b.terminal,
               suffix_trace=b.trace[meta[id(b)]['parent_turn']:]) for b in finished]
    return rows,dict(generated_tokens=tokens,executed_actions=actions,input_tokens=inputs,
                     seconds=time.monotonic()-start,continuations=len(rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['sft','sft_grpo400'],required=True)
    p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    cfg=json.loads((FROZEN/'configs/tree_random400.json').read_text())
    cfg['model']=str(OLD/('runs/golden_sft/sft/final_model' if a.model=='sft' else 'runs/formal/grpo_seed42/final_model'))
    all_tasks=json.loads((FROZEN/'artifacts/data.json').read_text())['train']
    batches=[[all_tasks[i] for i in np.random.default_rng(np.random.SeedSequence([42,b+1,1701])).choice(len(all_tasks),64,replace=False)] for b in range(3)]
    if a.audit_only:
        assert all(t['split']=='train' for batch in batches for t in batch)
        print(json.dumps({'batches':3,'tasks_each':64,'only_training_tasks':True}));return
    assert os.environ.get('NEXT_GPU_LEASE_FD');os.fstat(int(os.environ['NEXT_GPU_LEASE_FD']))
    out=ROOT/'runs/b1'/a.model;out.mkdir(parents=True,exist_ok=True)
    core.atomic_json(out/'identity.json',dict(model=cfg['model'],config=cfg,task_batches=batches,
        leaf_cap=8,K=[1,2,3,5],methods=['patr','tree_random'],shadow_replicates=2,
        shadow_scope='first decision with expanded and unexpanded parents per task; no further branching; same remaining horizon; diagnostic only, separately charged',
        selection_scope='training tasks only; no gradients or weight saves'))
    from train import build_models
    actor,ref,engine,tokenizer=build_models(cfg,42)
    original=core.route_patr
    for k in [1,2,3,5]:
        for method in ['patr','tree_random']:
            for bi,tasks in enumerate(batches):
                dest=out/f'k{k}_{method}_batch{bi+1}'
                if dest.with_suffix('.json').exists():continue
                assert time.time()<json.loads((ROOT/'artifacts/deadline.json').read_text())['unix']
                local=dict(cfg,branch_interval=k)
                router=BudgetRouter(core.score_branch,cfg['initial_branches'],8,capture=True)
                core.route_patr=router
                start=time.monotonic()
                try:
                    branches,stats,events=core.collect(tasks,engine,tokenizer,local,method,
                                                      np.random.default_rng(20261031+bi))
                finally:core.route_patr=original
                stats['seconds']=time.monotonic()-start
                groups={i:[b for b in branches if b.task_id==i] for i in range(64)}
                assert all(0<len(bs)<=8 for bs in groups.values())
                rows=[dict(task=tasks[i],leaf_count=len(bs),mixed=len({b.reward>0 for b in bs})>1,
                           successes=sum(b.reward>0 for b in bs)) for i,bs in groups.items()]
                shadow,cost=shadow_rollouts(router,tasks,engine,tokenizer,local,bi)
                with dest.with_suffix('.jsonl').open('w') as f:
                    for b in branches:f.write(json.dumps(dict(task=tasks[b.task_id],lineage=b.lineage,reward=b.reward,terminal=b.terminal,trace=b.trace))+'\n')
                forks=[e for e in events if e['kind']=='fork_candidates' and any(c['selected'] for c in e['candidates'])]
                result=dict(model=a.model,K=k,method=method,batch=bi+1,groups=rows,
                    actual_cost=stats,shadow_cost=cost,shadow=shadow,events=events,
                    forked_tasks=len({e['task_id'] for e in events if e['kind']=='expand'}),
                    fork_points=len(forks),multi_parent_fork_points=sum(len(e['candidates'])>1 for e in forks),
                    tied_fork_points=sum(len({c['score'] for c in e['candidates']})==1 for e in forks),
                    tied_multi_parent_forks=sum(len(e['candidates'])>1 and len({c['score'] for c in e['candidates']})==1 for e in forks),
                    score_prunes=sum(e.get('score_prune',False) for e in events),
                    mixed_groups=sum(r['mixed'] for r in rows))
                core.atomic_json(dest.with_suffix('.json'),result)
                core.atomic_json(out/'status.json',dict(status='running',K=k,method=method,batch=bi+1,updated_unix=time.time()))
    core.atomic_json(out/'summary.json',dict(status='complete',model=a.model,completed_cells=24,finished_unix=time.time()))

if __name__=='__main__':main()
