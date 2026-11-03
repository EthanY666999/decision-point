import ast,json,pathlib,tempfile,types,unittest
from unittest.mock import Mock
from recovery_utils import transient_image_error,safe_image_restart
class RecoveryTest(unittest.TestCase):
 def test_transient_and_permanent(self):
  for s in ['failed to fetch anonymous token: EOF','TLS handshake timeout','502 Bad Gateway']:self.assertTrue(transient_image_error(s))
  for s in ['manifest unknown','pull access denied','404 Client Error','unauthorized: bad credentials','reference patch does not pass']:self.assertFalse(transient_image_error(s))
 def test_only_pre_container_failure_can_restart(self):
  with tempfile.TemporaryDirectory() as d:
   r=pathlib.Path(d);p=r/'episodes/0052/episode.json';p.parent.mkdir(parents=True)
   e={'phase':'preparing','steps':[],'token':'abc'};p.write_text(json.dumps(e))
   s={'state':'attention_required','error':'500 Server Error images/create: EOF'}
   self.assertTrue(safe_image_restart(r,s))
   (r/'teacher_pending.json').write_text('{}');self.assertFalse(safe_image_restart(r,s));(r/'teacher_pending.json').unlink()
   q=r/'environment_states/abc/state.json';q.parent.mkdir(parents=True);q.write_text('{}');self.assertFalse(safe_image_restart(r,s));q.unlink()
   e['steps']=[{'command':'mutation'}];p.write_text(json.dumps(e));self.assertFalse(safe_image_restart(r,s))
 def test_actual_pull_retries_eof_and_preserves_event(self):
  self.check_pull(RuntimeError('anonymous token EOF'),True)
 def test_actual_pull_does_not_retry_denied(self):
  self.check_pull(RuntimeError('pull access denied'),False)
 def check_pull(self,error,should_retry):
  tree=ast.parse(pathlib.Path('remote_env.py').read_text())
  f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pull')
  class Missing(Exception):pass
  with tempfile.TemporaryDirectory() as d:
   clock=[0]
   def sleep(n):clock[0]+=n
   ns={'check_space':lambda:None,'docker':types.SimpleNamespace(errors=types.SimpleNamespace(ImageNotFound=Missing)),'ROOT':pathlib.Path(d),'time':types.SimpleNamespace(time=lambda:clock[0],sleep=sleep),'json':json}
   exec(compile(ast.Module(body=[f],type_ignores=[]),'remote_env.py','exec'),ns)
   c=types.SimpleNamespace(images=types.SimpleNamespace(get=Mock(side_effect=Missing()),pull=Mock(side_effect=[error,'image'])))
   if should_retry:
    self.assertEqual(ns['pull'](c,'pinned:image'),('image',True));self.assertEqual(c.images.pull.call_count,2);self.assertGreaterEqual(clock[0],5)
   else:
    with self.assertRaises(RuntimeError):ns['pull'](c,'pinned:image')
    self.assertEqual(c.images.pull.call_count,1)
   events=[json.loads(x) for x in (pathlib.Path(d)/'image_pull_events.jsonl').read_text().splitlines()]
   self.assertEqual(events[0]['retry'],should_retry)
if __name__=='__main__':unittest.main()
