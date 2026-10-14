"""Real-file schema4 execution-dependency and recovery regressions."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
import test_run_integrity as integrity_fixture
import run_integrity as integrity


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    base = tmp_path/'installed'
    distributions = {}
    for package, selected in integrity.RUNTIME_SOURCE_FILES.items():
        info = base/(package+'-1.0.dist-info')
        info.mkdir(parents=True)
        (info/'METADATA').write_text('Metadata-Version: 2.1\nName: '+package+'\nVersion: 1.0\n')
        paths = [info.name+'/METADATA', info.name+'/RECORD']+selected
        (info/'RECORD').write_text(''.join(path+',,\n' for path in paths))
        for relative in selected:
            path=base/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'actual fixture runtime bytes A\n')
        distributions[package]=importlib.metadata.PathDistribution(info)
    original=importlib.metadata.distribution
    monkeypatch.setattr(integrity.importlib.metadata,'distribution',lambda name:distributions[name] if name in distributions else original(name))
    fixture=integrity_fixture.RunIntegrityTests()
    fixture.setUp()
    yield fixture,base,distributions
    fixture.doCleanups()


def bytes_in(run):
    return {str(path.relative_to(run)):path.read_bytes() for path in run.rglob('*') if path.is_file()}


def test_selected_scope_is_exactly_approved_packages_and_files():
    assert set(integrity.RUNTIME_SOURCE_FILES)=={'vllm','transformers','torch','triton','xformers','tokenizers','numpy','gymnasium'}
    assert sum(map(len,integrity.RUNTIME_SOURCE_FILES.values()))==41
    assert 'vllm/engine/llm_engine.py' in integrity.RUNTIME_SOURCE_FILES['vllm']
    assert {'xformers/_C.so','xformers/_C_flashattention3.so'}<=set(integrity.RUNTIME_SOURCE_FILES['xformers'])
    assert 'tokenizers/tokenizers.abi3.so' in integrity.RUNTIME_SOURCE_FILES['tokenizers']


@pytest.mark.parametrize('relative',['vllm/engine/llm_engine.py','xformers/_C.so','tokenizers/tokenizers.abi3.so'])
def test_same_version_same_metadata_same_size_mutation_refuses_resume(runtime,relative):
    fixture,base,distributions=runtime
    fixture.prepare();fixture.checkpoint();fixture.logs()
    protected=bytes_in(fixture.run)
    metadata_before={str(path):path.read_bytes() for path in base.glob('*.dist-info/*')}
    target=base/relative
    stat=target.stat()
    target.write_bytes(target.read_bytes().replace(b'bytes A',b'bytes B'))
    os.utime(target,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    assert target.stat().st_size==stat.st_size and target.stat().st_mtime_ns==stat.st_mtime_ns
    assert metadata_before=={str(path):path.read_bytes() for path in base.glob('*.dist-info/*')}
    changed=fixture.fingerprint()
    assert changed['schema_version']==4 and changed['installed_dependencies']==fixture.fp['installed_dependencies']
    assert changed['sha256']!=fixture.fp['sha256']
    with pytest.raises(ValueError,match='Frozen source'):
        fixture.prepare(resume=True,fp=changed)
    assert bytes_in(fixture.run)==protected


@pytest.mark.parametrize('relative',['vllm-1.0.dist-info/RECORD','vllm-1.0.dist-info/METADATA'])
def test_actual_distribution_metadata_bytes_are_frozen(runtime,relative):
    fixture,base,_=runtime
    target=base/relative
    # Keep RECORD syntactically valid: importlib.metadata rejects blank CSV rows.
    suffix='vllm/engine/llm_engine.py,,\n' if target.name=='RECORD' else '\n'
    target.write_text(target.read_text()+suffix)
    changed=fixture.fingerprint()
    assert changed['installed_dependencies']==fixture.fp['installed_dependencies']
    assert changed['sha256']!=fixture.fp['sha256']


def test_installed_package_required_source_missing_is_explicit_failure(runtime):
    fixture,base,_=runtime
    (base/'vllm/engine/llm_engine.py').unlink()
    with pytest.raises(ValueError,match='Required installed runtime file missing'):
        fixture.fingerprint()


def test_runtime_hash_detects_in_read_change(runtime,monkeypatch):
    fixture,base,_=runtime
    target=base/'vllm/engine/llm_engine.py'
    original=integrity.sha256_file
    def racing(path):
        result=original(path)
        if Path(path)==target:
            target.write_bytes(target.read_bytes()+b'changed during read\n')
        return result
    monkeypatch.setattr(integrity,'sha256_file',racing)
    with pytest.raises(ValueError,match='changed during SHA256 read'):
        fixture.fingerprint()


def test_install_prefix_does_not_change_runtime_fingerprint(runtime,tmp_path,monkeypatch):
    fixture,base,_=runtime
    destination=tmp_path/'same-install-another-prefix'
    shutil.copytree(base,destination)
    monkeypatch.setattr(integrity.importlib.metadata,'distribution',
        lambda name:importlib.metadata.PathDistribution(destination/(name+'-1.0.dist-info')))
    assert fixture.fingerprint()==fixture.fp


def test_schema3_is_rejected_without_rewriting_any_historical_artifact(runtime):
    fixture,_,_=runtime
    legacy={key:value for key,value in fixture.fp.items() if key not in {'sha256','runtime_execution'}}
    legacy['schema_version']=3
    legacy['installed_dependencies']={key:value for key,value in legacy['installed_dependencies'].items() if key not in {'xformers','tokenizers'}}
    legacy['sha256']=integrity._digest(legacy)
    fixture.prepare(fp=legacy);fixture.checkpoint();fixture.logs()
    protected=bytes_in(fixture.run)
    with pytest.raises(ValueError,match='historical run metadata cannot be upgraded'):
        fixture.prepare(resume=True,fp=fixture.fp)
    assert bytes_in(fixture.run)==protected
    assert json.loads((fixture.run/'fingerprint.json').read_text())['schema_version']==3


@pytest.mark.parametrize('metrics_state,raw_state',[
    ('complete','partial'),('partial','complete'),('complete','complete'),('missing','complete')])
def test_final_checkpoint_resume_archives_uncommitted_final_attempt(metrics_state,raw_state):
    fixture=integrity_fixture.RunIntegrityTests();fixture.setUp()
    try:
        fixture.prepare();fixture.checkpoint(step=3);fixture.logs((1,2,3))
        (fixture.run/'status.json').write_text('{"status":"paused_checkpointed","step":3}')
        (fixture.run/'validation_0003.json').write_bytes(b'committed primary endpoint metrics')
        (fixture.run/'validation_0003.jsonl').write_bytes(b'committed primary endpoint raw\n')
        metric=fixture.run/'final_validation.json';raw=metric.with_suffix('.jsonl')
        if metrics_state!='missing':
            metric.write_bytes(b'{"seconds":12.5,"generated_tokens":456}' if metrics_state=='complete' else b'{"seconds":12.')
        raw.write_bytes(b'{"trajectory":"complete"}\n' if raw_state=='complete' else b'{"trajectory":"cut')
        prior={p.name:p.read_bytes() for p in (metric,raw) if p.exists()}
        protected={name:(fixture.run/name).read_bytes() for name in
            ('status.json','config.json','fingerprint.json','checkpoint.pt','train_metrics.jsonl',
             'branch_events.jsonl','validation_0003.json','validation_0003.jsonl')}
        result=fixture.prepare(resume=True)
        assert result['start_step']==3 and result['recovery'] is not None
        incident=result['recovery'];archive=Path(incident['archive'])
        assert {name:(archive/name).read_bytes() for name in prior}==prior
        assert all(not (fixture.run/name).exists() for name in prior)
        assert {name:(fixture.run/name).read_bytes() for name in protected}==protected
        assert incident['discarded_steps']==[] and incident['rolled_back_generated_tokens']==0
        assert incident['discarded_evaluation_total_cost_known'] is False
        assert incident['discarded_evaluation_seconds_recorded']==(12.5 if metrics_state=='complete' else 0)
        assert incident['discarded_evaluation_generated_tokens_recorded']==(456 if metrics_state=='complete' else 0)
        item=incident['archived_evaluations'][0]
        assert item['stem']=='final_validation' and item['raw_present']
        assert item['metrics_json_readable']==(metrics_state=='complete')
        # A later retry may create live output, but cannot replace archived bytes.
        metric.write_text('{"new":"retry"}');raw.write_text('{"new":"retry"}\n')
        assert {name:(archive/name).read_bytes() for name in prior}==prior
    finally:
        fixture.doCleanups()
