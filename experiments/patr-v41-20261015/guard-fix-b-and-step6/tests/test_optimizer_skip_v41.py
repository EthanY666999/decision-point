import unittest
import torch
from training_contract import Counters,optimizer_step
class SkipRule(unittest.TestCase):
    def test_finite_large_gradient_is_skipped_without_adam_state(self):
        p=torch.nn.Parameter(torch.tensor([1.]))
        optimizer=torch.optim.AdamW([p],lr=1e-6)
        counts=Counters()
        p.grad=torch.tensor([10.])
        norm,updated=optimizer_step(optimizer,[p],counts,torch)
        self.assertEqual(norm,10.);self.assertFalse(updated)
        self.assertEqual(p.item(),1.);self.assertEqual(len(optimizer.state),0)
        self.assertEqual(counts.skipped_updates,1)
        p.grad=torch.tensor([9.])
        norm,updated=optimizer_step(optimizer,[p],counts,torch)
        self.assertTrue(updated);self.assertNotEqual(p.item(),1.)
        self.assertEqual(counts.optimizer_updates,1)
