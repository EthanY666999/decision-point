import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_cert import normalize_rollout_stats
class RecordAdapterTest(unittest.TestCase):
 def test_diagnostics_preserved_without_collision(self):
  stats=dict(sampled_surprisal=1.25,generated_tokens=19,probe_cost={"generated_tokens":2})
  changes=dict(sampled_surprisal=1.5,grad_norm=.2)
  merged=dict(**normalize_rollout_stats(stats),**changes)
  self.assertEqual(merged["sampled_surprisal"],1.5)
  self.assertEqual(merged["rollout_sampled_surprisal"],1.25)
  self.assertEqual(merged["probe_cost"],stats["probe_cost"])
  self.assertIn("sampled_surprisal",stats)
if __name__=="__main__":unittest.main()
