"""CPU behavioral checks; never imports torch, vLLM or the frozen runtime."""
from types import SimpleNamespace
import copy
import numpy as np
from b2_collect import make_collect
from branch_budget import BudgetRouter

class Branch(SimpleNamespace):
    def clone(self, suffix):
        b = copy.deepcopy(self); b.lineage += suffix; return b

def initial():
    return [Branch(task_id=0, turns=2, lineage=str(i), order=(i,), actions=[i],
                   reward=float(i % 2), score=1, terminal=None) for i in range(4)]

cfg = dict(branch_interval=2, leaf_cap=8, initial_branches=4, branch_factor=2,
           top_k_branches=2, prune_margin=.5, repeat_limit=6)
sentinel = object()
core = SimpleNamespace(route_patr=sentinel, score_branch=lambda b, c: b.score)
def original(tasks, engine, tokenizer, config, method, rng, evaluation=False):
    if evaluation:
        assert core.route_patr is sentinel
        return ['untouched evaluation'], {}, []
    active = initial(); events = []
    for _ in range(4):
        active, done, ev = core.route_patr(active, config, rng, method == 'tree_random')
        assert not done
        events += [dict(e, task_id=0) for e in ev]
    assert not core.route_patr.snapshots
    return active, {'generated_tokens': 123}, events

collect = make_collect(core, original)
for method in ['patr', 'tree_random']:
    for _ in range(2):
        branches, stats, events = collect([{}], None, None, cfg, method, np.random.default_rng(42))
        assert len(branches) == 8 and stats['mixed_groups'] == 1 and stats['max_leaves'] == 8
        assert stats['generated_tokens'] == 123 and core.route_patr is sentinel
        assert any(e['kind'] == 'cap_continue' for e in events)
assert collect([{}], None, None, cfg, 'grpo', None, evaluation=True)[0] == ['untouched evaluation']
def fail(*a, **kw): raise RuntimeError('injected collection failure')
try: make_collect(core, fail)([{}], None, None, cfg, 'patr', None)
except RuntimeError: pass
else: raise AssertionError('Expected failure')
assert core.route_patr is sentinel
# Dropping terminal leaves from active work never refunds their budget.
router = BudgetRouter(core.score_branch, cap=8)
active, _, _ = router(initial(), cfg, np.random.default_rng(1))
assert router.leaves[0] == 6
active = active[2:]
active, _, _ = router(active, cfg, np.random.default_rng(1))
assert router.leaves[0] == 8 and len(active) == 6
active, _, _ = router(active, cfg, np.random.default_rng(1))
assert router.leaves[0] == 8 and len(active) == 6
print('PASS: fresh batch budgets, both selectors, cap continuation, terminal consumption, no shadow, evaluation bypass, exception restoration')
