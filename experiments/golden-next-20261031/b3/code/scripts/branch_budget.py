"""Explicit cumulative leaf cap. No CUDA imports; used only in new experiments."""
import copy
import numpy as np

class BudgetRouter:
    def __init__(self, score, initial=4, cap=8, capture=False):
        assert cap >= initial > 0
        self.score, self.initial, self.cap, self.capture = score, initial, cap, capture
        self.leaves = {}
        self.snapshots = {}

    def __call__(self, active, cfg, rng, random_selection=False):
        if not active:return [], [], []
        assert cfg['branch_factor']==2, 'This cap implementation is binary only'
        ti=active[0].task_id
        assert all(b.task_id==ti for b in active)
        count=self.leaves.setdefault(ti,self.initial)
        scores=np.array([self.score(b,cfg) for b in active])
        ties=rng.permutation(len(active))
        order=ties if random_selection else ties[np.argsort(-scores[ties],kind='stable')]
        nominated=set(order[:min(cfg['top_k_branches'],len(active))].tolist())
        expand=set(order[:min(cfg['top_k_branches'],len(active),self.cap-count)].tolist())
        self.leaves[ti]+=len(expand)
        assert self.leaves[ti]<=self.cap
        if self.capture and expand and len(expand)<len(active) and ti not in self.snapshots:
            # First comparable decision per task. Shadow continuations never enter RL.
            self.snapshots[ti]=[(copy.deepcopy(b),i in expand,float(scores[i])) for i,b in enumerate(active)]
        median=float(np.median(scores))
        keep,done=[],[]
        events=[dict(kind='fork_candidates',turn=active[0].turns,leaf_count_before=count,
            leaf_count_after=self.leaves[ti],leaf_cap=self.cap,
            candidates=[dict(lineage=b.lineage,score=float(scores[i]),selected=i in expand,
                             nominated=i in nominated,cap_blocked=i in nominated and i not in expand)
                        for i,b in enumerate(active)])]
        for i,b in enumerate(active):
            if i in expand:
                for j in range(2):
                    child=b.clone(f'.{j}');child.order=b.order+(j,);keep.append(child)
                events.append(dict(kind='expand',lineage=b.lineage,score=float(scores[i]),turn=b.turns))
            elif i in nominated:
                # A selected branch denied extra leaves continues normally.
                keep.append(b)
                events.append(dict(kind='cap_continue',lineage=b.lineage,turn=b.turns))
            else:
                score_prune=not random_selection and scores[i]<median-cfg['prune_margin']
                repeat=len(b.actions)>=cfg['repeat_limit'] and len(set(b.actions[-cfg['repeat_limit']:]))==1
                if score_prune or repeat:
                    b.terminal='pruned';done.append(b)
                    events.append(dict(kind='prune',lineage=b.lineage,score=float(scores[i]),turn=b.turns,
                                       score_prune=bool(score_prune),repeat_prune=bool(repeat)))
                else:keep.append(b)
        return keep,done,events
