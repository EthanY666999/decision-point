import unittest
import training_contract as tc

class V43(unittest.TestCase):
    def test_zero_variance_groups_kept_with_zero_advantage(self):
        rows, zero = tc.filter_and_advantages([[0,0,0],[1,1],[0,1,1,1]])
        self.assertEqual(zero, [0,1])
        self.assertEqual(len(rows), 3+2+4)
        self.assertTrue(all(a == 0.0 for g,_,a in rows if g in (0,1)))
        mixed=[a for g,_,a in rows if g==2]
        self.assertAlmostEqual(sum(mixed), 0.0, places=9)
        self.assertTrue(any(a != 0 for a in mixed))
    def test_legacy_filter_still_available(self):
        rows, zero = tc.filter_and_advantages([[0,0],[0,1]], keep_zero_variance=False)
        self.assertEqual(len(rows), 2); self.assertEqual(zero, [0])
    def test_default_is_keep(self):
        self.assertTrue(tc.KEEP_ZERO_VARIANCE)
    def test_more_than_256_rows_gives_two_minibatches(self):
        rows=[(i//6,i%6,0.) for i in range(64*6)]
        self.assertEqual([len(m) for m in tc.trajectory_minibatches(rows)],[256,128])

if __name__=='__main__':unittest.main()
