import unittest
from monitoring_policy import partition
from guards_v41 import StabilityGuard
class Policy(unittest.TestCase):
 def test_large_finite_gradient_and_restore(self):
  g=StabilityGuard(lengths=[50]*10,gradients=[.87]*20,seconds=[60]*20)
  g=StabilityGuard(**g.state_dict())
  fatal,warnings=partition(g.check(50,0,[210.6,7.6],60))
  self.assertEqual(fatal,[]);self.assertEqual(warnings,['gradient_above_20x'])
 def test_all_statistics_alert(self):
  g=StabilityGuard(lengths=[50]*10,gradients=[1]*20,seconds=[60]*20)
  fatal,warnings=partition(g.check(500,.1,[100],1000))
  self.assertEqual(fatal,[]);self.assertEqual(len(warnings),4)
 def test_nonfinite_stays_fatal(self):
  for v in [float('nan'),float('inf')]:
   fatal,warnings=partition(StabilityGuard().check(50,0,[v],60))
   self.assertEqual(fatal,['nonfinite_or_negative_monitor']);self.assertEqual(warnings,[])
 def test_unknown_stays_fatal(self):
  self.assertEqual(partition(['checkpoint_corrupt']),(['checkpoint_corrupt'],[]))
