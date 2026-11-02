"""CPU checks of the nontrivial cap semantics, independent of GPU/runtime."""
import copy
from types import SimpleNamespace
import numpy as np
from branch_budget import BudgetRouter

class Branch(SimpleNamespace):
    def clone(self,suffix):
        child=copy.deepcopy(self);child.lineage+=suffix;return child

def branches(scores):
    return [Branch(task_id=0,turns=1,lineage=f'r{i}',order=(i,),actions=[1],terminal=None,score=s) for i,s in enumerate(scores)]

cfg=dict(branch_factor=2,top_k_branches=2,prune_margin=.5,repeat_limit=6)
for random in [False,True]:
    route=BudgetRouter(lambda b,c:b.score,capture=True)
    active=branches([.8,.8,.8,.8]);rng=np.random.default_rng(42)
    for turn in range(1,10):
        before=len(active)
        active,done,events=route(active,cfg,rng,random)
        assert not done and len(active)<=8
        if turn>=3:assert len(active)==before==8 and any(e['kind']=='cap_continue' for e in events)
    assert route.leaves[0]==8 and len(route.snapshots)==1
# Score pruning remains PaTR-only and is explicitly labelled.
p=BudgetRouter(lambda b,c:b.score);r=BudgetRouter(lambda b,c:b.score)
_,done,events=p(branches([1,1,1,0]),cfg,np.random.default_rng(1),False)
assert len(done)==1 and any(e.get('score_prune') for e in events)
_,done,_=r(branches([1,1,1,0]),cfg,np.random.default_rng(1),True)
assert not done
# One spare leaf means one binary expansion; nominations denied budget continue.
p=BudgetRouter(lambda b,c:b.score,cap=5)
active,done,events=p(branches([1,1,1,0]),cfg,np.random.default_rng(1),False)
assert sum(e['kind']=='expand' for e in events)==1 and p.leaves[0]==5
assert any(e['kind']=='cap_continue' for e in events)
assert len(active)+len(done)==5
print('PASS: cumulative cap, terminal-inclusive count, blocked continuation, selection and pruning semantics')
