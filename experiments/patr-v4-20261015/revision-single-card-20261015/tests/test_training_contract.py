import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_contract import Counters, EpochSampler, filter_and_advantages, trajectory_minibatches


class ContractTests(unittest.TestCase):
    def test_complete_groups_and_tail(self):
        groups = [[0] * 8, [1] * 8] + [[0, 0, 1, 1, 0, 1]] * 45
        rows, rejected = filter_and_advantages(groups)
        self.assertEqual(rejected, [0, 1])
        self.assertEqual([len(x) for x in trajectory_minibatches(rows)], [256, 14])
        for group in range(2, 47):
            advantages = [r[2] for r in rows if r[0] == group]
            self.assertEqual(len(advantages), 6)
            self.assertAlmostEqual(sum(advantages), 0)

    def test_resume_crosses_epoch_identically(self):
        sampler = EpochSampler(130)
        sampler.next_batch()
        saved = sampler.state_dict()
        expected = [sampler.next_batch() for _ in range(7)]
        restored = EpochSampler(130, seed=999)
        restored.load_state_dict(saved)
        self.assertEqual(expected, [restored.next_batch() for _ in range(7)])
        self.assertEqual(saved['cursor'], 64)
        self.assertEqual(len(set(expected[1] + expected[2])), 128)
        corrupt = copy.deepcopy(saved)
        corrupt['permutation'][0] = corrupt['permutation'][1]
        with self.assertRaises(ValueError):
            restored.load_state_dict(corrupt)

    def test_rejected_batches_remain_in_cost(self):
        counts = Counters()
        for _ in range(19):
            self.assertTrue(counts.record_batch(64, 100, False))
        self.assertFalse(counts.record_batch(64, 100, False))
        self.assertEqual(counts.accepted_iteration, 0)
        self.assertEqual(counts.sampled_tasks, 1280)
        self.assertEqual(counts.sampled_tokens, 2000)
        self.assertTrue(counts.record_batch(64, 120, True))
        self.assertEqual(counts.accepted_iteration, 1)
        self.assertEqual(counts.consecutive_empty, 0)

    def test_nonfinite_rewards_rejected(self):
        with self.assertRaises(ValueError):
            filter_and_advantages([[0, float('nan')]])


if __name__ == '__main__':
    unittest.main()
