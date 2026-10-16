import unittest, math
import torch
from loss_v41 import token_mean
from training_contract import Counters, optimizer_step, SKIP_THRESHOLD
import train_v4

class V42(unittest.TestCase):
    def test_token_mean_equal_weights_and_micro_equivalence(self):
        v=torch.tensor([2.,4.,9.,9.,9.],dtype=torch.float64,requires_grad=True)
        full=token_mean(v,5)
        self.assertAlmostEqual(full.item(),33/5)
        micro=token_mean(v[:2],5)+token_mean(v[2:],5)
        torch.testing.assert_close(full,micro,rtol=0,atol=1e-12)
        full.backward()
        torch.testing.assert_close(v.grad,torch.full((5,),.2,dtype=torch.float64))
    def test_long_loop_not_downweighted(self):
        # A 10240-token loop and a 4-token answer: every token weight is identical.
        v=torch.ones(10244,dtype=torch.float64,requires_grad=True)
        token_mean(v,10244).backward()
        self.assertEqual(v.grad.max().item(),v.grad.min().item())
    def test_reduction_weights_sum_to_one(self):
        # Total weight is always 1 regardless of length (no v4 token-sum blow-up).
        for n in (4,400,40000):
            v=torch.ones(n,dtype=torch.float64,requires_grad=True)
            token_mean(v,n).backward(); self.assertAlmostEqual(v.grad.sum().item(),1.)
    def test_invalid_denominator(self):
        with self.assertRaises(ValueError): token_mean(torch.ones(3),2)
    def test_no_skip_at_rllm_default(self):
        self.assertEqual(SKIP_THRESHOLD,1e5)
        p=torch.nn.Parameter(torch.tensor([1.]));opt=torch.optim.AdamW([p],lr=1e-6);c=Counters()
        p.grad=torch.tensor([15.]); norm,updated=optimizer_step(opt,[p],c,torch)
        self.assertTrue(updated);self.assertEqual(c.skipped_updates,0)
    def test_nonfinite_still_hard_stop(self):
        p=torch.nn.Parameter(torch.tensor([1.]));opt=torch.optim.AdamW([p],lr=1e-6)
        p.grad=torch.tensor([math.inf])
        with self.assertRaises(FloatingPointError): optimizer_step(opt,[p],Counters(),torch)
    def test_fence_loop_detector(self):
        loop={'response':'```Left``` ``` ``` ``` ``` ``` ``` ``` ```','response_tokens':40}
        ok={'response':'```Left```','response_tokens':4}
        reason={'response':'P is above G. ```Down```','response_tokens':12}
        self.assertEqual(train_v4.fence_loop_fraction([loop,ok,reason,ok]),.25)
        self.assertEqual(train_v4.fence_loop_fraction([]),0)

if __name__=='__main__':unittest.main()
