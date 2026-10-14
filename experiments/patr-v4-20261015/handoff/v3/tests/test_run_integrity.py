"""Crash recovery and immutable provenance tests using real temporary files."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_integrity as integrity


def json_save(state, path):
    Path(path).write_text(json.dumps(state))


def json_load(path):
    return json.loads(Path(path).read_text())


class RunIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'v2'
        self.root.mkdir()
        (self.root / 'artifacts/model').mkdir(parents=True)
        (self.root / 'vendor').mkdir()
        (self.root / 'train.py').write_text('VALUE = 1\n')
        (self.root / 'vendor/env.py').write_text('LIMIT = 10\n')
        (self.root / 'requirements.txt').write_text('numpy==1.0\n')
        self.data = self.root / 'artifacts/data.json'
        self.data.write_text('{"train": [], "validation": [], "test": []}')
        model_file = self.root / 'artifacts/model/config.json'
        model_file.write_text('{"architecture": "test"}')
        weight = self.root / 'artifacts/model/model.safetensors'
        weight.write_bytes(b'test weight')
        manifest = {'files': [dict(path=p.name, bytes=p.stat().st_size, sha256=integrity.sha256_file(p)) for p in (model_file, weight)]}
        integrity.atomic_json(self.root / 'artifacts/model_manifest.json', manifest)
        self.cfg = dict(model='artifacts/model', method='grpo', steps=3, seed=42,
                        micro_batch_size=2, vllm_attention_backend='FLASH_ATTN')
        self.run = self.root / 'runs/short/grpo_seed42'
        self.fp = self.fingerprint()

    def fingerprint(self, cfg=None):
        return integrity.build_fingerprint(cfg or self.cfg, self.data, root=self.root)

    def prepare(self, resume=False, fp=None, cfg=None):
        return integrity.prepare_run(self.run, resume, fp or self.fp, cfg or self.cfg,
                                     root=self.root, checkpoint_loader=json_load)

    def checkpoint(self, step=1):
        integrity.atomic_checkpoint(self.run / 'checkpoint.pt', dict(step=step, config=self.cfg), save_fn=json_save)

    def logs(self, steps=(1,)):
        for name in ('train_metrics.jsonl', 'branch_events.jsonl'):
            records = [dict(step=i, generated_tokens=100 * i, input_tokens=200 * i,
                            step_seconds=10 * i, events=[]) for i in steps]
            (self.run / name).write_text(''.join(json.dumps(record) + '\n' for record in records))

    def test_source_vendor_and_dependency_changes_rejected_before_metadata_write(self):
        self.prepare()
        self.checkpoint()
        self.logs()
        original = (self.run / 'config.json').read_bytes()
        for rel in ('train.py', 'vendor/env.py', 'requirements.txt'):
            path = self.root / rel
            content = path.read_text()
            path.write_text(content + '\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source'):
                self.prepare(resume=True, fp=self.fingerprint())
            self.assertEqual((self.run / 'config.json').read_bytes(), original)
            path.write_text(content)

    def test_config_change_does_not_overwrite_original_metadata(self):
        self.prepare()
        self.checkpoint()
        self.logs()
        original = (self.run / 'config.json').read_bytes()
        changed = dict(self.cfg, steps=5)
        with self.assertRaisesRegex(ValueError, 'Frozen source'):
            self.prepare(resume=True, fp=self.fingerprint(changed), cfg=changed)
        self.assertEqual((self.run / 'config.json').read_bytes(), original)

    def test_model_manifest_tokenizer_and_data_changes_detected(self):
        original = self.fp['sha256']
        self.data.write_text('{"train": [1]}')
        self.assertNotEqual(self.fingerprint()['sha256'], original)
        config_file = self.root / 'artifacts/model/config.json'
        config_file.write_text(config_file.read_text().replace('test', 'fake'))
        with self.assertRaisesRegex(ValueError, 'differs from manifest'):
            self.fingerprint()

    def test_full_weight_hash_cache_reuses_only_unchanged_file_identity(self):
        cache = self.root / 'artifacts/model_sha256_cache.json'
        cache.unlink()
        original_hash = integrity.sha256_file
        with patch.object(integrity, 'sha256_file', wraps=original_hash) as hashed:
            self.fingerprint()
            self.assertEqual(sum(Path(call.args[0]).suffix == '.safetensors' for call in hashed.call_args_list), 1)
        with patch.object(integrity, 'sha256_file', wraps=original_hash) as hashed:
            self.fingerprint()
            self.assertEqual(sum(Path(call.args[0]).suffix == '.safetensors' for call in hashed.call_args_list), 0)
        weight = self.root / 'artifacts/model/model.safetensors'
        stat = weight.stat()
        os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        with patch.object(integrity, 'sha256_file', wraps=original_hash) as hashed:
            self.fingerprint()
            self.assertEqual(sum(Path(call.args[0]).suffix == '.safetensors' for call in hashed.call_args_list), 1)
        self.assertEqual(json_load(cache)['model.safetensors']['verification'], 'full_file_sha256')

    def test_same_size_weight_corruption_is_rejected_even_with_mtime_restored(self):
        weight = self.root / 'artifacts/model/model.safetensors'
        before = weight.stat()
        weight.write_bytes(b'fake weight')
        os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'differs from manifest'):
            self.fingerprint()

    def test_missing_model_never_creates_or_writes_model_target(self):
        model = self.root / 'artifacts/model'
        model.rename(self.root / 'artifacts/saved-model')
        model.symlink_to(self.root.parent / 'missing-model-target', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Model file missing'):
            self.fingerprint()
        self.assertFalse((self.root.parent / 'missing-model-target').exists())

    def test_existing_run_requires_resume_and_resume_requires_checkpoint(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'already contains evidence'):
            self.prepare()
        with self.assertRaisesRegex(ValueError, 'requires checkpoint'):
            self.prepare(resume=True)
        self.assertEqual(json_load(self.run / 'config.json'), self.cfg)

    def test_completed_run_resume_preserves_final_evaluation_and_metadata(self):
        self.prepare()
        self.checkpoint(step=3)
        self.logs((1, 2, 3))
        (self.run / 'status.json').write_text('{"status":"complete","step":3}')
        (self.run / 'final_validation.json').write_text('{"success_rate_pct":42.0}')
        (self.run / 'final_validation.jsonl').write_text('{"raw":"original endpoint"}\n')
        protected = ('status.json', 'config.json', 'fingerprint.json', 'checkpoint.pt',
                     'final_validation.json', 'final_validation.jsonl', 'train_metrics.jsonl')
        original = {name: (self.run / name).read_bytes() for name in protected}
        with self.assertRaisesRegex(ValueError, 'Completed run cannot be resumed'):
            self.prepare(resume=True)
        self.assertEqual({name: (self.run / name).read_bytes() for name in protected}, original)
        self.assertFalse((self.run / 'recovery_history').exists())

    def test_numerical_environment_change_rejected_before_metadata_write(self):
        self.prepare()
        self.checkpoint()
        self.logs()
        original = (self.run / 'config.json').read_bytes()
        new_value = ':16:8' if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':16:8' else ':4096:8'
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': new_value}):
            changed = self.fingerprint()
            self.assertNotEqual(changed['sha256'], self.fp['sha256'])
            with self.assertRaisesRegex(ValueError, 'Frozen source'):
                self.prepare(resume=True, fp=changed)
        self.assertEqual((self.run / 'config.json').read_bytes(), original)

    def test_torch_determinism_flag_change_changes_fingerprint(self):
        self.prepare()
        self.checkpoint()
        self.logs()
        environment = dict(self.fp['numerical_environment'])
        flags = dict(environment.get('torch_flags') or {})
        flags['deterministic_algorithms'] = not flags.get('deterministic_algorithms', False)
        environment['torch_flags'] = flags
        with patch.object(integrity, 'numerical_environment', return_value=environment):
            changed = self.fingerprint()
            with self.assertRaisesRegex(ValueError, 'Frozen source'):
                self.prepare(resume=True, fp=changed)

    def test_runtime_capacity_override_is_recorded_without_rewriting_config(self):
        self.prepare()
        self.checkpoint()
        self.logs()
        changed = dict(self.cfg, micro_batch_size=1)
        fp = self.fingerprint(changed)
        self.assertEqual(fp, self.fp)
        self.prepare(resume=True, fp=fp, cfg=changed)
        self.assertEqual(json_load(self.run / 'config.json'), self.cfg)
        self.assertEqual(json_load(self.run / 'runtime_override_current.json')['micro_batch_size'], 1)
        eager = dict(self.cfg, vllm_enforce_eager=True)
        with self.assertRaisesRegex(ValueError, 'Frozen source'):
            self.prepare(resume=True, fp=self.fingerprint(eager), cfg=eager)
        changed['vllm_attention_backend'] = 'XFORMERS'
        with self.assertRaisesRegex(ValueError, 'Frozen source'):
            self.prepare(resume=True, fp=self.fingerprint(changed), cfg=changed)

    def test_recovery_archives_every_uncommitted_artifact_and_cost(self):
        self.prepare()
        self.checkpoint(step=1)
        self.logs((1, 2, 3))
        for name in ('rollouts_0001.jsonl.gz', 'rollouts_0002.jsonl.gz',
                     'dapo_candidates_0003.jsonl.gz', 'validation_0002.json',
                     'validation_0002.jsonl', 'final_validation.json', 'final_validation.jsonl'):
            (self.run / name).write_bytes(b'saved evidence')
        parent = self.root.parent / 'legacy_result.json'
        parent.write_bytes(b'legacy must remain intact')
        result = self.prepare(resume=True)
        self.assertEqual(result['start_step'], 1)
        incident = result['recovery']
        self.assertEqual(incident['rolled_back_generated_tokens'], 500)
        self.assertEqual(incident['rolled_back_step_seconds'], 50)
        archive = Path(incident['archive'])
        self.assertEqual([r['step'] for r in integrity.read_step_records(self.run / 'train_metrics.jsonl')], [1])
        self.assertEqual(len(integrity.read_step_records(archive / 'train_metrics.jsonl')), 3)
        self.assertTrue((self.run / 'rollouts_0001.jsonl.gz').exists())
        for name in incident['files']:
            self.assertFalse((self.run / name).exists())
            self.assertTrue((archive / name).exists())
        self.assertEqual(parent.read_bytes(), b'legacy must remain intact')
        self.assertIsNone(self.prepare(resume=True)['recovery'])

    def test_torn_tail_is_preserved_and_unknown_cost_explicit(self):
        self.prepare()
        self.checkpoint(step=1)
        self.logs((1,))
        path = self.run / 'train_metrics.jsonl'
        with path.open('ab') as stream:
            stream.write(b'{"step": 2, "generated_tokens":')
        result = self.prepare(resume=True)
        incident = result['recovery']
        self.assertFalse(incident['incomplete_iteration_cost_known'])
        self.assertEqual(incident['torn_append_files'], ['train_metrics.jsonl'])
        self.assertTrue((Path(incident['archive']) / 'train_metrics.jsonl').read_bytes().endswith(b'"generated_tokens":'))
        self.assertEqual(len(integrity.read_step_records(path)), 1)

    def test_duplicate_step_rejected_not_last_record_wins(self):
        self.prepare()
        self.checkpoint(step=1)
        self.logs((1, 2, 2))
        before = (self.run / 'train_metrics.jsonl').read_bytes()
        with self.assertRaisesRegex(ValueError, 'duplicate step'):
            integrity.read_step_records(self.run / 'train_metrics.jsonl')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.prepare(resume=True)
        self.assertEqual((self.run / 'train_metrics.jsonl').read_bytes(), before)
        self.assertFalse((self.run / 'recovery_history').exists())

    def test_missing_committed_history_fails_closed(self):
        self.prepare()
        self.checkpoint(step=2)
        self.logs((1, 3))
        with self.assertRaisesRegex(ValueError, 'missing training metrics'):
            self.prepare(resume=True)

    def test_failed_checkpoint_save_preserves_previous_checkpoint(self):
        self.prepare()
        self.checkpoint(step=1)
        before = (self.run / 'checkpoint.pt').read_bytes()
        def fail(state, path):
            Path(path).write_bytes(b'partial checkpoint')
            raise OSError('out of disk')
        with self.assertRaisesRegex(OSError, 'out of disk'):
            integrity.atomic_checkpoint(self.run / 'checkpoint.pt', dict(step=2), save_fn=fail)
        self.assertEqual((self.run / 'checkpoint.pt').read_bytes(), before)
        self.assertEqual(list(self.run.glob('.checkpoint.*.tmp')), [])

    def test_foreign_checkpoint_and_parent_directory_rejected(self):
        self.prepare()
        self.checkpoint(step=1)
        state = json_load(self.run / 'checkpoint.pt')
        state['integrity_fingerprint'] = 'wrong'
        json_save(state, self.run / 'checkpoint.pt')
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            self.prepare(resume=True)
        with self.assertRaisesRegex(ValueError, 'parent/legacy'):
            integrity.prepare_run(self.root.parent / 'legacy/run', False, self.fp, self.cfg, root=self.root)

    def test_report_uses_same_strict_duplicate_reader(self):
        spec = importlib.util.spec_from_file_location('v2_report', Path(__file__).resolve().parents[1] / 'scripts/report.py')
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        self.prepare()
        self.logs((1, 1))
        with self.assertRaisesRegex(ValueError, 'duplicate step'):
            report.records(self.run / 'train_metrics.jsonl')


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('gymnasium'), 'environment replay dependencies unavailable')
class RawEvaluationEvidenceTests(unittest.TestCase):
    def setUp(self):
        from scripts import verify_final
        self.verify = verify_final
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'validation_0001.json'
        self.cfg = dict(max_turns=10, max_response_tokens=100, eval_samples=4, is_slippery=False)
        self.task = dict(uid='fixture-map', seed=42, size=2, p=.8, desc=['SG', 'FF'])
        env = verify_final.make_env(self.task, self.cfg)
        observation, reward, done, _ = env.step(3)
        self.records = []
        for lineage in range(4):
            self.records.append(dict(task=self.task, lineage=str(lineage), reward=1., terminal='success',
                turns=1, response_tokens=3, ids=[99, 10, 11, 2], mask=[0, 1, 1, 1],
                old_logprobs=[0., -.2, -.3, -.1], trace=[dict(action=3, response='```Right```',
                state=int(env.s), observation=observation, reward=reward,
                response_token_ids=[10, 11, 2], response_tokens=3)]))
        self.metrics = dict(success_rate_pct=100., pass_at_4_pct=100., response_length=3.,
            response_length_excluding_eos=2., turns=1., invalid_action_fraction=0.,
            first_turn_invalid_fraction=0., n_tasks=1, n_trajectories=4,
            terminal_counts={'success': 4}, generated_tokens=12, split='validation', eos_token_id=2)
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.metrics))
        self.path.with_suffix('.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in self.records))

    def test_valid_raw_tokens_and_environment_replay(self):
        result = self.verify.audit_evaluation(self.path, [self.task], self.cfg, 'validation')
        self.assertTrue(result['raw_token_accounting_verified'])
        self.assertEqual(result['metrics']['response_length_excluding_eos'], 2.)

    def test_counter_and_trace_token_mismatches_are_rejected(self):
        self.records[0]['response_tokens'] = 2
        self.write()
        with self.assertRaisesRegex(ValueError, 'RLen does not equal'):
            self.verify.audit_evaluation(self.path, [self.task], self.cfg, 'validation')
        self.records[0]['response_tokens'] = 3
        self.records[0]['trace'][0]['response_token_ids'][0] = 15
        self.write()
        with self.assertRaisesRegex(ValueError, 'Trace response IDs differ'):
            self.verify.audit_evaluation(self.path, [self.task], self.cfg, 'validation')

    def test_first_response_truncation_counts_as_invalid(self):
        self.metrics.update(success_rate_pct=75., pass_at_4_pct=100., turns=.75,
                            first_turn_invalid_fraction=.25, terminal_counts={'success': 3, 'token_limit': 1})
        record = self.records[0]
        record.update(reward=0., terminal='token_limit', turns=0)
        record['trace'][0].update(action=None, reward=0., terminal='token_limit')
        self.write()
        result = self.verify.audit_evaluation(self.path, [self.task], self.cfg, 'validation')
        self.assertEqual(result['metrics']['first_turn_invalid_fraction'], .25)


if __name__ == '__main__':
    unittest.main()
