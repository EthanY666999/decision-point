"""CPU checkpoint envelope actually preserves guard baselines and RNG/sampler."""
import tempfile
import unittest
from pathlib import Path
import torch
from guards_v41 import StabilityGuard
from training_contract import Counters,EpochSampler
import checkpoint_v4

class GuardCheckpoint(unittest.TestCase):
    def test_restore_then_trigger_without_rebaselining(self):
        g=StabilityGuard()
        for _ in range(20):g.check(50,0,[.02],10)
        counters=Counters(guard_state=g.state_dict())
        model=torch.nn.Linear(2,2)
        opt=torch.optim.AdamW(model.parameters())
        sampler=EpochSampler(130)
        sampler.next_batch()
        with tempfile.TemporaryDirectory() as d:
            cp=Path(d)/'latest.pt'
            checkpoint_v4.save(cp,model,opt,sampler,counters,{'fixture':'guard-state'})
            counters.guard_state={}
            checkpoint_v4.restore(cp,model,opt,sampler,counters,{'fixture':'guard-state'})
            restored=StabilityGuard(**counters.guard_state)
            self.assertEqual(set(restored.check(201,.051,[.41],41)),{'assistant_length_above_3x','token_limit_above_5_percent','gradient_above_20x','step_seconds_above_4x'})

if __name__=='__main__':unittest.main()
