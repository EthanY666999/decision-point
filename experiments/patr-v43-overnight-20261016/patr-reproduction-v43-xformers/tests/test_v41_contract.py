import unittest
import torch
from loss_v41 import low_var_kl, trajectory_mean
from guards_v41 import StabilityGuard

class Contract(unittest.TestCase):
    def test_equal_trajectory_weight_and_micro_equivalence(self):
        values=torch.tensor([2.,4.,9.,9.,9.],dtype=torch.float64,requires_grad=True)
        full=trajectory_mean(values,[2,3],2)
        self.assertEqual(full.item(),6.)
        micro=trajectory_mean(values[:2],[2],2)+trajectory_mean(values[2:],[3],2)
        torch.testing.assert_close(full,micro,rtol=0,atol=0)
        full.backward()
        torch.testing.assert_close(values.grad,torch.tensor([.25,.25,1/6,1/6,1/6],dtype=torch.float64))
    def test_actual_tail_denominator(self):
        self.assertEqual(trajectory_mean(torch.ones(3),[1,2],2).item(),1.)
    def test_kl_clamps_and_gradient(self):
        logp=torch.tensor([-100.,-2.,0.,2.,100.],dtype=torch.float64,requires_grad=True)
        out=low_var_kl(logp,torch.zeros_like(logp))
        expected=torch.tensor([10.,torch.exp(torch.tensor(2.)).item()-3,0.,torch.exp(torch.tensor(-2.)).item()+1,10.],dtype=torch.float64)
        torch.testing.assert_close(out,expected,rtol=1e-6,atol=1e-6)
        out.sum().backward();self.assertTrue(torch.isfinite(logp.grad).all())
        self.assertEqual(logp.grad[0].item(),0.);self.assertEqual(logp.grad[-1].item(),0.)
    def test_guard_bootstrap_threshold_and_restore(self):
        g=StabilityGuard()
        for _ in range(20):self.assertEqual(g.check(50,.05,[.01],60),[])
        self.assertEqual(g.check(200,.05,[.2],240),[])
        restored=StabilityGuard(**g.state_dict())
        expected={'assistant_length_above_3x','token_limit_above_5_percent','gradient_above_20x','step_seconds_above_4x'}
        self.assertEqual(set(restored.check(201,.051,[.201],241)),expected)
    def test_early_absolute_guard(self):
        self.assertEqual(StabilityGuard().check(1,.051,[],1),['token_limit_above_5_percent'])
    def test_nan(self):
        self.assertEqual(StabilityGuard().check(float('nan'),0,[],1),['nonfinite_or_negative_monitor'])

if __name__=='__main__':unittest.main()
