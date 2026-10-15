"""Atomic checkpoint envelope; integration must supply complete frozen identity."""
import os
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch


def checksum(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def metadata_path(path):
    return Path(str(path)+'.manifest.json')


def atomic_metadata(path, value):
    temp=Path(str(path)+'.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,sort_keys=True)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def verified_load(path):
    manifest=json.loads(metadata_path(path).read_text())
    if Path(path).stat().st_size!=manifest['bytes'] or checksum(path)!=manifest['sha256']:
        raise ValueError('Checkpoint bytes differ from committed SHA256 manifest')
    return torch.load(path,map_location='cpu',weights_only=False)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def save(path, model, optimizer, sampler, counters, fingerprint):
    if not fingerprint:
        raise ValueError('Missing frozen fingerprint')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # FP32 weights and two FP32 Adam moments, plus conservative overhead.
    required = sum(p.numel()*p.element_size() for p in model.parameters())*3 + 5*1024**3
    if shutil.disk_usage(path.parent).free < required:
        raise OSError('Insufficient space for atomic checkpoint and 4GiB reserve')
    payload = dict(version=4, fingerprint=fingerprint, model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state_dict(), counters=counters.state_dict(), rng=capture_rng())
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    manifest=dict(bytes=tmp.stat().st_size,sha256=checksum(tmp),version=4)
    # Keep one previous committed generation across the two-file commit window.
    # Hardlinks retain bytes without copying the existing large checkpoint.
    if path.exists():
        previous=Path(str(path)+'.previous')
        old_manifest=json.loads(metadata_path(path).read_text())
        if path.stat().st_size!=old_manifest['bytes'] or checksum(path)!=old_manifest['sha256']:
            raise ValueError('Existing checkpoint failed integrity; refusing overwrite')
        link=Path(str(previous)+'.tmp')
        if link.exists():link.unlink()
        os.link(path,link)
        os.replace(link,previous)
        atomic_metadata(metadata_path(previous),old_manifest)
    os.replace(tmp, path)
    atomic_metadata(metadata_path(path),manifest)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def restore(path, model, optimizer, sampler, counters, fingerprint):
    # Only locally generated, trusted checkpoints may be loaded here.
    path=Path(path)
    used=path
    try:
        state=verified_load(path)
    except (OSError,ValueError,KeyError,json.JSONDecodeError,RuntimeError) as error:
        used=Path(str(path)+'.previous')
        if not used.exists():raise
        state=verified_load(used)
        atomic_metadata(Path(str(path)+'.recovery.json'),dict(recovered_from=str(used),reason=repr(error)))
    if state.get('version') != 4 or state.get('fingerprint') != fingerprint:
        raise ValueError('Checkpoint source/config/data/model fingerprint mismatch')
    if set(state['counters']) != set(vars(counters)):
        raise ValueError('Checkpoint counter schema mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    sampler.load_state_dict(state['sampler'])
    for key, value in state['counters'].items():
        setattr(counters, key, value)
    restore_rng(state['rng'])
    if used!=path:
        # Preserve the damaged generation as evidence, then repair the current
        # pointer so a later save need not overwrite or silently discard it.
        suffix='.invalid-'+str(time.time_ns())
        if path.exists():os.replace(path,Path(str(path)+suffix))
        if metadata_path(path).exists():
            os.replace(metadata_path(path),Path(str(metadata_path(path))+suffix))
        link=Path(str(path)+'.repair')
        if link.exists():link.unlink()
        os.link(used,link)
        os.replace(link,path)
        atomic_metadata(metadata_path(path),json.loads(metadata_path(used).read_text()))
    return dict(path=str(used),fallback=used!=path)
