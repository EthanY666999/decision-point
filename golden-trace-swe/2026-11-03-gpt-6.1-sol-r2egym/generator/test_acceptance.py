"""Acceptance invariants: an empty, partial or timed-out test run is never gold."""
import ast
import hashlib
import json
import pathlib
import tempfile
import unittest

ROOT=pathlib.Path(__file__).resolve().parent

class AcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree=ast.parse((ROOT/'remote_env.py').read_text())
        wanted={'grade','safe_action_paths'}
        cls.ns={'pathlib':pathlib,'json':json,'sha':lambda b:hashlib.sha256(b).hexdigest(),
                'save':lambda p,x:p.write_text(json.dumps(x)),
                'decolor_dict_keys':lambda d:d,'parse_log_fn':lambda repo:json.loads}
        module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted],type_ignores=[])
        exec(compile(module,'remote_env.py','exec'),cls.ns)

    def grade(self,actual,expected,code=0):
        self.ns['run']=lambda *args:{'output':json.dumps(actual),'exit_code':code,'seconds':.1}
        with tempfile.TemporaryDirectory() as d:
            return self.ns['grade'](None,{'repo_name':'x','expected_output_json':json.dumps(expected)},
                {'test_command':'unused'},pathlib.Path(d),'check')['passed']

    def test_empty_never_passes(self): self.assertFalse(self.grade({},{}))
    def test_exact_statuses(self): self.assertTrue(self.grade({'a':'PASSED'},{'a':'PASSED'}))
    def test_partial_never_passes(self): self.assertFalse(self.grade({'a':'PASSED'},{'a':'PASSED','b':'PASSED'}))
    def test_failure_never_passes(self): self.assertFalse(self.grade({'a':'FAILED'},{'a':'PASSED'}))
    def test_timeout_never_passes(self): self.assertFalse(self.grade({'a':'PASSED'},{'a':'PASSED'},124))
    def test_test_patch_rejected(self):
        for p in ['tests/test_a.py','a/test_b.py','a/b_test.py','pytest.ini','../host.py','/tmp/file.py','.git/config']:
            self.assertFalse(self.ns['safe_action_paths']([p]),p)
        self.assertTrue(self.ns['safe_action_paths'](['aiohttp/client.py']))

if __name__=='__main__': unittest.main()
