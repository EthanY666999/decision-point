"""B2-only sampling adapter. No CUDA imports or shadow continuations."""
from branch_budget import BudgetRouter


def make_collect(core, original):
    def collect(tasks, engine, tokenizer, cfg, method, rng, evaluation=False):
        if evaluation:
            return original(tasks, engine, tokenizer, cfg, method, rng, evaluation=True)
        assert method in ('patr', 'tree_random')
        assert cfg['branch_interval'] == 2 and cfg['leaf_cap'] == 8
        assert cfg['initial_branches'] == 4 and cfg['branch_factor'] == 2
        router = BudgetRouter(core.score_branch, initial=4, cap=8, capture=False)
        previous = core.route_patr
        core.route_patr = router
        try:
            branches, stats, events = original(tasks, engine, tokenizer, cfg, method, rng)
        finally:
            core.route_patr = previous
        groups = [[b for b in branches if b.task_id == ti] for ti in range(len(tasks))]
        assert all(0 < len(bs) <= 8 for bs in groups)
        assert all(len(bs) == router.leaves.get(ti, 4) for ti, bs in enumerate(groups))
        assert not router.snapshots, 'Shadow states must never enter training'
        forks = [e for e in events if e['kind'] == 'fork_candidates' and any(c['selected'] for c in e['candidates'])]
        mixed = sum(len({b.reward > 0 for b in bs}) > 1 for bs in groups)
        stats = dict(stats, mixed_groups=mixed, task_groups=len(tasks),
                     mixed_group_fraction=mixed / len(tasks), leaf_cap=8,
                     max_leaves=max(map(len, groups)),
                     forked_tasks=len({e['task_id'] for e in events if e['kind'] == 'expand'}),
                     fork_points=len(forks),
                     multi_parent_fork_points=sum(len(e['candidates']) > 1 for e in forks),
                     tied_multi_parent_forks=sum(len(e['candidates']) > 1 and len({c['score'] for c in e['candidates']}) == 1 for e in forks),
                     score_prunes=sum(bool(e.get('score_prune')) for e in events))
        return branches, stats, events
    return collect
