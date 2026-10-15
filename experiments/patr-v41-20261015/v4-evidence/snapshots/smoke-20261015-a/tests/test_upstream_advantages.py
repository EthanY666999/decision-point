import ast
from collections import defaultdict
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training_contract import filter_and_advantages


class UpstreamAdvantageTests(unittest.TestCase):
    def test_variable_complete_groups_match_pinned_function(self):
        source=Path(__file__).resolve().parents[1]/'reference/upstream_core_algos.py'
        tree=ast.parse(source.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_grpo_outcome_advantage')
        node.decorator_list=[]
        ns=dict(torch=torch,np=np,defaultdict=defaultdict)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned_advantage','exec'),ns)
        groups=[[0.,1.],[0.,0.,0.,1.,1.,1.],[0.]*7+[1.],[0.]*13+[1.],[0.]*8,[1.]*6]
        rows,rejected=filter_and_advantages(groups)
        self.assertEqual(rejected,[4,5])
        rewards=torch.tensor([groups[g][j] for g,j,a in rows]).reshape(-1,1)
        indices=np.array([g for g,j,a in rows])
        expected=ns['compute_grpo_outcome_advantage'](rewards,torch.ones_like(rewards),indices)[0].flatten()
        actual=torch.tensor([a for g,j,a in rows])
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
