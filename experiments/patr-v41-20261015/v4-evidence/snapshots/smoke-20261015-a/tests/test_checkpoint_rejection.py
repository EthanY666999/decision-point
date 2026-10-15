import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_v4 import save, restore, checksum, atomic_metadata, metadata_path, verified_load
from training_contract import EpochSampler, Counters


class CheckpointRejectionTests(unittest.TestCase):
    def setUp(self):
        self.get_cuda=patch('checkpoint_v4.torch.cuda.get_rng_state_all',return_value=[])
        self.set_cuda=patch('checkpoint_v4.torch.cuda.set_rng_state_all')
        self.get_cuda.start();self.set_cuda.start()
        self.addCleanup(self.get_cuda.stop);self.addCleanup(self.set_cuda.stop)

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
            atomic_metadata(metadata_path(path),dict(bytes=path.stat().st_size,sha256=checksum(path),version=3))
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                restore(path, model, optimizer, sampler, counters, identity)
            torch.testing.assert_close(model.weight, modified, atol=0, rtol=0)
            self.assertFalse(torch.equal(saved, modified))

    def test_damaged_latest_uses_verified_previous_generation(self):
        model=torch.nn.Linear(2,1)
        optimizer=torch.optim.AdamW(model.parameters())
        sampler,counts=EpochSampler(128),Counters()
        identity={'source':'fixed'}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.pt'
            first=model.weight.detach().clone()
            save(path,model,optimizer,sampler,counts,identity)
            with torch.no_grad():model.weight.add_(2)
            counts.accepted_iteration=1
            save(path,model,optimizer,sampler,counts,identity)
            with path.open('ab') as stream:stream.write(b'corruption')
            result=restore(path,model,optimizer,sampler,counts,identity)
            self.assertTrue(result['fallback'])
            torch.testing.assert_close(model.weight,first,atol=0,rtol=0)
            self.assertEqual(counts.accepted_iteration,0)
            self.assertEqual(verified_load(path)['counters']['accepted_iteration'],0)
            self.assertTrue(list(Path(directory).glob('latest.pt.invalid-*')))
            save(path,model,optimizer,sampler,counts,identity)

    def test_corrupt_only_checkpoint_is_rejected(self):
        model=torch.nn.Linear(2,1)
        optimizer=torch.optim.AdamW(model.parameters())
        sampler,counts=EpochSampler(128),Counters()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.pt'
            save(path,model,optimizer,sampler,counts,{'source':'fixed'})
            with path.open('ab') as stream:stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                restore(path,model,optimizer,sampler,counts,{'source':'fixed'})
