import unittest
from recovery_guard import next_state
class Bounds(unittest.TestCase):
 def setUp(self):
  self.s=dict(total=0,ram_original_restart=0,cuda_original_restart=0,history=[])
  self.e=dict(cause='verified resource fault',corrective_action='auxiliary load removed',resource_improvement_verified=True,capacity_sufficient=True,deadline_unix=2000)
 def test_persistent_total_and_per_kind(self):
  a=next_state(self.s,'cuda',self.e,0)
  with self.assertRaises(ValueError):next_state(a,'cuda',self.e,0)
  b=next_state(a,'ram',self.e,0)
  with self.assertRaises(ValueError):next_state(b,'disk',self.e,0)
  self.assertEqual(self.s['total'],0)
 def test_no_blind_retry(self):
  for k in ['resource_improvement_verified','capacity_sufficient']:
   with self.assertRaises(ValueError):next_state(self.s,'ram',dict(self.e,**{k:False}),0)
 def test_deadline(self):
  with self.assertRaises(ValueError):next_state(self.s,'disk',self.e,1100)
if __name__=='__main__':unittest.main()
