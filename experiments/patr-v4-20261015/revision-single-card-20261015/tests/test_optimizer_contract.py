import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_contract import Counters, optimizer_step, scaled_microbatch_loss


class OptimizerTests(unittest.TestCase):
    def test_tail_and_microbatch_gradients(self):
        for n in (6, 12, 66, 78, 256, 270):
            x = torch.arange(n, dtype=torch.float64, requires_grad=True)
            reference = (x / 10240).mean() * (n / 256)
            expected, = torch.autograd.grad(reference, x)
            loss = sum(scaled_microbatch_loss(part) for part in x.split(7))
            actual, = torch.autograd.grad(loss, x)
            torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-12)

    def test_skipped_step_preserves_adam_state(self):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        opt = torch.optim.AdamW([p], lr=1e-6)
        counts = Counters()
        p.grad = torch.tensor([1.0])
        self.assertTrue(optimizer_step(opt, [p], counts, torch)[1])
        before = p.detach().clone()
        state = {k: v.clone() for k, v in opt.state[p].items()}
        p.grad = torch.tensor([10.0])
        self.assertFalse(optimizer_step(opt, [p], counts, torch)[1])
        torch.testing.assert_close(p, before, atol=0, rtol=0)
        for k, value in state.items():
            torch.testing.assert_close(opt.state[p][k], value, atol=0, rtol=0)
        p.grad = torch.tensor([float('nan')])
        with self.assertRaises(FloatingPointError):
            optimizer_step(opt, [p], counts, torch)
        self.assertEqual((counts.optimizer_attempts, counts.optimizer_updates, counts.skipped_updates), (3, 1, 2))


if __name__ == '__main__':
    unittest.main()
