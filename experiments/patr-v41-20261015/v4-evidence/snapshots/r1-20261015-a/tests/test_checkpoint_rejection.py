import sys
import tempfile
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_v4 import save, restore
from training_contract import EpochSampler, Counters


class CheckpointRejectionTests(unittest.TestCase):
    def test_identity_and_schema_fail_closed_before_weights_load(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        sampler, counters = EpochSampler(128), Counters()
        identity = dict(source='A', data='B', config='C', model='D')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            save(path, model, optimizer, sampler, counters, identity)
            saved = model.weight.detach().clone()
            with torch.no_grad():
                model.weight.add_(5)
            modified = model.weight.detach().clone()
            for key in identity:
                wrong = dict(identity, **{key: 'changed'})
                with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                    restore(path, model, optimizer, sampler, counters, wrong)
                torch.testing.assert_close(model.weight, modified, atol=0, rtol=0)
            state = torch.load(path, weights_only=False)
            state['version'] = 3
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                restore(path, model, optimizer, sampler, counters, identity)
            torch.testing.assert_close(model.weight, modified, atol=0, rtol=0)
            self.assertFalse(torch.equal(saved, modified))
