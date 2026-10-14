import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trajectory_budget import TrajectoryBudget


class BudgetTests(unittest.TestCase):
    def test_terminal_observation_counted_but_not_stored(self):
        b = TrajectoryBudget(10)
        self.assertEqual(b.append_turn([1, 2], [1, 1], [3, 4], [0, 0], 1., True), (1., True, False))
        self.assertEqual((b.counted_tokens, b.tokens, b.mask), (4, [1, 2], [1, 1]))

    def test_environment_overflow_preserves_success(self):
        b = TrajectoryBudget(5)
        self.assertEqual(b.append_turn([1, 2], [1, 1], [3, 4, 5, 6], [0]*4, 1., True), (1., True, True))
        self.assertEqual(b.mask, [1, 1, 0, 0, 0])

    def test_assistant_overflow_clears_reward(self):
        b = TrajectoryBudget(5)
        b.append_turn([1], [1], [2, 3], [0, 0], 0., False)
        self.assertEqual(b.append_turn([4, 5, 6], [1]*3, [], [], 1., True), (0., True, True))
        self.assertEqual(b.tokens, [1, 2, 3, 4, 5])

    def test_exact_boundary_preserves_reward(self):
        b = TrajectoryBudget(2)
        self.assertEqual(b.append_turn([1, 2], [1, 1], [], [], 1., True), (1., True, True))

    def test_environment_policy_mask_is_error(self):
        with self.assertRaises(ValueError):
            TrajectoryBudget().append_turn([1], [1], [2], [1], 0., False)
