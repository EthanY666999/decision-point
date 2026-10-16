"""Execute production monitor construction/call, not a hand-written substitute."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from guards_v41 import StabilityGuard
from monitoring_policy import partition

class DriverMonitor(unittest.TestCase):
    def test_actual_driver_keyword_contract(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'train_v4.py').read_text())
        nodes=[]
        for name in ('monitor','reasons'):
            found=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for target in n.targets for t in ast.walk(target))]
            self.assertEqual(len(found),1)
            nodes.extend(found)
        ns=dict(branches=[SimpleNamespace(ntokens=300,terminal='token_limit'),SimpleNamespace(ntokens=10,terminal='success')],result={'gradient_norms':[.1]},record={'seconds':60},guard=StabilityGuard(),partition=partition)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_driver_monitor','exec'),ns)
        self.assertEqual(ns['monitor']['mean_assistant_tokens'],155)
        self.assertEqual(ns['reasons'],[])
        self.assertEqual(ns['warnings'],['token_limit_above_5_percent'])
        ns['result']={'gradient_norms':[float('nan')]}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_driver_nan','exec'),ns)
        self.assertEqual(ns['reasons'],['nonfinite_or_negative_monitor'])
        json.dumps(ns['guard'].state_dict(),allow_nan=False)
