import unittest
from alerts_v43 import advance
class Alerts(unittest.TestCase):
 def test_ten_and_reset(self):
  s={};d={'first_action_top_share':.91,'optimization':{'accepted_groups':2},'step':1,'raw_batch':0}
  for _ in range(9):self.assertEqual(advance(s,d),[])
  self.assertEqual(len(advance(s,d)),2)
  self.assertEqual(advance(s,d),[])
  d['first_action_top_share']=.9;d['optimization']['accepted_groups']=3
  advance(s,d);self.assertTrue(all(v==0 for v in s.values()))
 def test_fence_three(self):
  s={};d={'first_action_top_share':.1,'optimization':{'accepted_groups':64},'step':1,'raw_batch':0,'fence_loop_fraction':.02}
  self.assertEqual(advance(s,d),[]);self.assertEqual(advance(s,d),[])
  self.assertEqual(advance(s,d)[0]['action'],'report_only')
