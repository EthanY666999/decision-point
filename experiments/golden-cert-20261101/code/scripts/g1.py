"""Predeclared paired CERT sampling. No optimizer, validation, test or weight writes."""
import argparse,copy,gzip,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import patr_core as core
from cert_collect import initial,prepare_nodes,continuations,node_record
from cert_select import SELECTORS,select
from cert_entropy import token_map


def leaf_row(b,tasks):
    return dict(task=tasks[b.task_id],task_id=b.task_id,lineage=b.lineage,reward=b.reward,terminal=b.terminal,ids=b.ids,mask=b.mask,old_logprobs=b.old_logprobs,trace=b.trace)

def write_gz(p,rows):
    tmp=p.with_suffix(p.suffix+'.tmp')
    with gzip.open(tmp,'wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    tmp.replace(p)

def compare(a,b):
    def sig(xs):return {(x.task_id,x.lineage):(x.ids,x.mask,x.old_logprobs,x.actions,x.reward,x.terminal) for x in xs}
    return sig(a)==sig(b)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['sft','sft_grpo400','sft_patr400'],required=True);a=p.parse_args()
    assert os.environ.get('NEXT_GPU_LEASE_FD');os.fstat(int(os.environ['NEXT_GPU_LEASE_FD']))
    protocol=json.loads((ROOT/'artifacts/protocol.json').read_text());cfg=json.loads((ROOT/'configs/cert_g2_uct.json').read_text())
    cfg['model']=protocol['models'][a.model]
    cfg['_deadline_unix']=json.loads((ROOT/'artifacts/deadline.json').read_text())['unix']
    out=ROOT/'runs/g1'/a.model;out.mkdir(parents=True,exist_ok=True)
    data=json.loads((ROOT/'artifacts/data.json').read_text())['train']
    batches=[[data[i] for i in np.random.default_rng(np.random.SeedSequence([42,b+1,1701])).choice(len(data),64,replace=False)] for b in range(3)]
    for name,sha in {**protocol['source_sha256'],**protocol['config_sha256']}.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    assert json.loads((ROOT/'artifacts/cpu_acceptance.json').read_text())['passed']
    identity=dict(model=a.model,model_path=cfg['model'],task_batches=batches,config=cfg,protocol=protocol)
    if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==identity
    else:core.atomic_json(out/'identity.json',identity)
    # Frozen builder loads actor/reference but never constructs an optimizer.
    from train import build_models
    actor,ref,engine,tokenizer=build_models(cfg,42)
    core.atomic_json(out/'token_mapping.json',token_map(tokenizer)[1])
    os.environ['PATR_PROGRESS_PATH']=str(out/'progress.json')
    for bi,tasks in enumerate(batches):
        dest=out/f'batch{bi+1}.json'
        if dest.exists():continue
        base=int(np.random.default_rng(20261031+bi).integers(0,2**31-1))
        if bi==0 and not (out/'gpu_consistency.json').exists():
            comparisons={};costs={};start=time.monotonic()
            for topk in (0,1,20):
                leaves,_,cost,_=initial(core,tasks,engine,tokenizer,cfg,base,topk)
                comparisons[topk]=leaves;costs[str(topk)]=cost
                write_gz(out/f'consistency_top{topk}.jsonl.gz',[leaf_row(b,tasks) for b in leaves])
            passed=compare(comparisons[0],comparisons[20]) and compare(comparisons[1],comparisons[20])
            core.atomic_json(out/'gpu_consistency.json',dict(passed=passed,exact_token_ids_mask_old_logprobs_actions_reward_terminal=passed,logprobs=[0,1,20],cost=costs,seconds=time.monotonic()-start))
            if not passed:
                core.atomic_json(out/'status.json',dict(status='consistency_failed',no_training=True));raise RuntimeError('GPU consistency failed: preserve evidence and stop CERT')
            del comparisons
        assert json.loads((out/'gpu_consistency.json').read_text())['passed']
        first,nodes,icost,_=initial(core,tasks,engine,tokenizer,cfg,base,20)
        candidates,ph=prepare_nodes(nodes,first,tasks);plans={};union={ti:{} for ti in candidates}
        for forks in (1,2):
            for selector in SELECTORS:
                chosen={ti:select(ns,selector,forks,ph[ti],base,tasks[ti]['uid'],bi+1) for ti,ns in candidates.items()}
                plans[(forks,selector)]=chosen
                for ti,ns in chosen.items():
                    for n in ns:union[ti][(n['j'],n['turn'])]=n
        added,ccost,skips=continuations(core,{ti:sorted(ns.values(),key=lambda n:(n['j'],n['turn'])) for ti,ns in union.items()},tasks,engine,tokenizer,cfg,base)
        write_gz(out/f'batch{bi+1}_initial.jsonl.gz',[leaf_row(b,tasks) for b in first])
        write_gz(out/f'batch{bi+1}_continuations.jsonl.gz',[leaf_row(x[0],tasks) for x in added.values()])
        arms={}
        for (forks,selector),chosen in plans.items():
            rows=[];cost={k:icost[k] for k in ('generated_tokens','input_tokens','executed_actions','invalid_actions','request_seconds')}
            for ti,ns in chosen.items():
                leaves=[b for b in first if b.task_id==ti];new=[];dup=[];selected=[]
                for n in ns:
                    key=(ti,n['j'],n['turn']);selected.append([n['j'],n['turn']])
                    if key not in added:continue
                    b,c,d=added[key];new.append(b)
                    if d is not None:dup.append(d)
                    for k in cost:cost[k]+=c[k]
                leaves+=new;initial_mixed=0<ph[ti]<1;mixed=len({b.reward>0 for b in leaves})>1
                rows.append(dict(uid=tasks[ti]['uid'],p_hat=ph[ti],initial_mixed=initial_mixed,mixed=mixed,rescued=not initial_mixed and mixed,
                  initial_all_success=ph[ti]==1,initial_all_failure=ph[ti]==0,new_leaves=len(new),new_successes=sum(b.reward>0 for b in new),
                  duplicate_first=sum(dup),duplicate_denominator=len(dup),selected=selected,selected_turns=[n['turn'] for n in ns],
                  selected_H_cons=[n['H_cons'] for n in ns],unselected_H_cons=[n['H_cons'] for n in candidates[ti] if [n['j'],n['turn']] not in selected],
                  candidate_shortfall=len(ns)<forks))
            arms[f'F{forks}_{selector}']=dict(rows=rows,mixed_groups=sum(r['mixed'] for r in rows),cost=cost,
              timing_note='Exact per-arm tokens/actions including full initial cost; request_seconds sums overlapped latency, NOT wall time. Actual joint measured wall time is separately reported.')
        core.atomic_json(dest,dict(model=a.model,batch=bi+1,base_seed=base,arms=arms,initial_cost=icost,shared_union_continuation_cost=ccost,replay_skips=skips,
           candidates={tasks[ti]['uid']:[node_record(n) for n in ns] for ti,ns in candidates.items()},
           diagnostic=dict(nodes=len(nodes),format_missing=sum(n['diagnostics']['format_missing'] for n in nodes),action_first=sum(n['diagnostics']['action_first'] for n in nodes),action_first_parse_failed=sum(n['diagnostics']['action_first_parse_failed'] for n in nodes))))
        core.atomic_json(out/'status.json',dict(status='sampling',completed_batches=bi+1,total_batches=3,no_training=True))
    core.atomic_json(out/'status.json',dict(status='complete',completed_batches=3,no_training=True,finished=time.time()))

if __name__=='__main__':main()
