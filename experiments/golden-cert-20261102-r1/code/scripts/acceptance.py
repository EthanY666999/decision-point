"""Predeclared repeated frozen-production controls, observer/probe isolation and calibration."""
import argparse,gzip,json,os,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import patr_core as core
from cert_collect import initial
from cert_select import stable_seed
from g1 import write_gz,leaf_row
TERMINALS=('success','hole','turn_limit','token_limit','context_limit','pruned_repeat')
METRICS=('success','mean_turns',*('terminal_'+t for t in TERMINALS))
MARGINS=np.array([.05,.5]+[.05]*len(TERMINALS))

def vectors(leaves,tasks):
    rows=[]
    for ti,t in enumerate(tasks):
        bs=[b for b in leaves if b.task_id==ti];assert len(bs)==4
        assert all(len(b.ids)==len(b.mask)==len(b.old_logprobs) and all(np.isfinite(b.old_logprobs)) for b in bs)
        rows.append([sum(b.reward>0 for b in bs)/4,sum(b.turns for b in bs)/4]+[sum(b.terminal==term for b in bs)/4 for term in TERMINALS])
    return np.array(rows)

def equivalence(a,b,c,seed=20261102):
    # arrays [repeat, UID, metric]; cluster repeated outcomes by training UID.
    effect=(c-(a+b)/2).mean(axis=0);rng=np.random.default_rng(seed)
    sims=np.array([effect[rng.integers(0,len(effect),len(effect))].mean(axis=0) for _ in range(10000)])
    lo,hi=np.quantile(sims,[.05,.95],axis=0)
    baseline_envelope=np.max(np.abs((a-b).mean(axis=1)),axis=0)
    mean=effect.mean(axis=0)
    checks=(lo>-MARGINS)&(hi<MARGINS)
    return dict(passed=bool(checks.all()),metrics={k:dict(difference=float(mean[i]),ci90=[float(lo[i]),float(hi[i])],
      margin=float(MARGINS[i]),baseline_repeat_envelope=float(baseline_envelope[i]),passed=bool(checks[i])) for i,k in enumerate(METRICS)})

def calibration(rows):
    # One canonical node per prefix; each action uses the same complete node sample.
    by={}
    for r in rows:by.setdefault(r['uid'],[]).append([r['pi'][a]-int(r['actual']==a) for a in range(5)])
    sums=np.array([np.sum(v,axis=0) for v in by.values()]);counts=np.array([len(v) for v in by.values()]);rng=np.random.default_rng(20261103)
    sims=[]
    for _ in range(10000):
        ix=rng.integers(0,len(counts),len(counts));sims.append(sums[ix].sum(axis=0)/counts[ix].sum())
    lo,hi=np.quantile(sims,[.005,.995],axis=0);mean=sums.sum(axis=0)/counts.sum()
    return dict(passed=bool(((lo>-.10)&(hi<.10)).all()),nodes=len(rows),uid_clusters=len(by),margin=.10,
       familywise_note='99% intervals for five action proxies (Bonferroni family 95%)',
       actions={str(a):dict(mean_error=float(mean[a]),ci99=[float(lo[a]),float(hi[a])]) for a in range(5)},
       limitation='Specified-format first-token proxy against final parsed action; failure does not identify its cause')

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['sft','sft_grpo400','sft_patr400']);args=p.parse_args()
    os.fstat(int(os.environ['NEXT_GPU_LEASE_FD']))
    out=ROOT/'runs/acceptance'/args.model;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'result.json').exists(),'Never overwrite a completed acceptance; preserve failed evidence'
    protocol=json.loads((ROOT/'artifacts/protocol.json').read_text());cfg=json.loads((ROOT/'configs/cert_g2_uct.json').read_text());cfg['model']=protocol['models'][args.model]
    cfg['_deadline_unix']=json.loads((ROOT/'artifacts/deadline.json').read_text())['unix']
    from train import build_models
    actor,ref,engine,tok=build_models(cfg,42)
    data=json.loads((ROOT/'artifacts/data.json').read_text())['train'];tasks=[data[i] for i in np.random.default_rng(np.random.SeedSequence([42,1,1701])).choice(len(data),64,replace=False)]
    original=core.apply_response;arrays={k:[] for k in ('A','B','C')};cal=[];allcost=[];agreements=[]
    started=time.monotonic()
    for rep in range(6):
        base=stable_seed('cert-v2-acceptance',20261104,rep)
        # C = observed production after probes. Alternate order deterministically.
        order=('A','B','C') if rep%2==0 else ('C','B','A');runs={}
        for mode in order:
            if mode=='C':
                # A separate warmup rollout supplies real nodes; probe results and warmup
                # leaves do not join comparison/optimizer data. Cost is explicit.
                _,nodes,pc,_=initial(core,tasks,engine,tok,cfg,stable_seed(base,'probe_warmup'),with_probes=True)
                allcost.append(dict(repeat=rep,mode='probe_warmup',cost=pc))
                for n in nodes:cal.append(dict(uid=tasks[n['branch'].task_id]['uid'],pi=[n['pi'][a] for a in range(5)],actual=n['parent_action'] if n['parent_action'] in (1,2,3,4) else 0))
                write_gz(out/f'r{rep}_probes.jsonl.gz',[dict(uid=tasks[n['branch'].task_id]['uid'],j=n['j'],turn=n['turn'],pi=n['pi'],diagnostics=n['diagnostics'],actual=n['parent_action']) for n in nodes])
                leaves,_,cost,_=initial(core,tasks,engine,tok,cfg,base,with_probes=False)
            else:
                class FixedSeed:
                    def integers(self,*a,**kw):return base
                ts=time.monotonic();leaves,cost,_=core.collect(tasks,engine,tok,dict(cfg,branch_interval=11),'patr',FixedSeed());cost['seconds']=time.monotonic()-ts
            assert core.apply_response is original
            arrays[mode].append((rep,vectors(leaves,tasks)));runs[mode]=leaves
            allcost.append(dict(repeat=rep,mode=mode,cost=cost));write_gz(out/f'r{rep}_{mode}.jsonl.gz',[leaf_row(b,tasks) for b in leaves])
            core.atomic_json(out/'status.json',dict(status='running',repeat=rep+1,mode=mode,repeats=6,updated=time.time()))
        def agree(x,y):return dict(ids=sum(a.ids==b.ids for a,b in zip(x,y)),old_logprobs=sum(a.old_logprobs==b.old_logprobs for a,b in zip(x,y)),trajectories=len(x))
        agreements.append(dict(repeat=rep,A_B=agree(runs['A'],runs['B']),A_C=agree(runs['A'],runs['C'])))
    arr={k:np.array([v for _,v in sorted(xs,key=lambda x:x[0])]) for k,xs in arrays.items()}
    eq=equivalence(arr['A'],arr['B'],arr['C']);cl=calibration(cal)
    result=dict(passed=eq['passed'] and cl['passed'],production_equivalence=eq,calibration=cl,repeat_agreement=agreements,cost=allcost,
      seconds=time.monotonic()-started,no_training=True,no_test_use=True,old_exact_failure_preserved=True)
    core.atomic_json(out/'result.json',result);core.atomic_json(out/'status.json',dict(status='passed' if result['passed'] else 'failed',updated=time.time()))
    print(json.dumps({k:result[k] for k in ('passed','seconds')}))
if __name__=='__main__':main()
