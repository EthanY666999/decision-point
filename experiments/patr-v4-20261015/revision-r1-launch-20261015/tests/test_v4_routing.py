import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patr_core import Branch, route_patr, prune_repeated_response
from trajectory_budget import TrajectoryBudget


class RoutingTests(unittest.TestCase):
    def branch(self, i):
        env = SimpleNamespace(s=0, ncol=2, nrow=2, desc=np.array([list('SF'),list('FG')],dtype='S1'))
        return Branch(0,env,None,[7,8],[0,1],[0.,-.5],actions=[3]*5,turns=5,initial_distance=2,lineage=str(i),order=(i,),budget=TrajectoryBudget(10240,1,[8],[1]))

    def test_top_two_expand_shared_prefix_without_mutation(self):
        cfg=dict(scorer_reward_weight=1.,scorer_progress_weight=1.,patr_tie_break='random',top_k_branches=2,branch_factor=2,prune_margin=.5,repeat_limit=6)
        branches=[self.branch(i) for i in range(4)]
        leaves,done,events=route_patr(branches,cfg,np.random.default_rng(42))
        self.assertEqual((len(leaves),len(done),len(events)),(6,0,2))
        children=[b for b in leaves if '.' in b.lineage]
        self.assertEqual(len(children),4)
        children[0].budget.tokens.append(99)
        self.assertTrue(all(b.budget.tokens==[8] for b in branches))
        self.assertTrue(all(b.budget.tokens==[8] for b in children[1:]))

    def test_repeat_six_training_only(self):
        cfg=dict(repeat_limit=6,repeat_check_every_turn=True)
        b=self.branch(0)
        self.assertIsNone(prune_repeated_response(b,cfg,'patr',False))
        b.actions.append(3);b.turns=6
        self.assertIsNone(prune_repeated_response(b,cfg,'patr',True))
        self.assertIsNone(prune_repeated_response(b,cfg,'grpo',False))
        self.assertEqual(prune_repeated_response(b,cfg,'patr',False)['kind'],'prune_repeat')
        self.assertEqual(b.terminal,'pruned')
