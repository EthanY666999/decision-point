"""Atomic checkpoint envelope; integration must supply complete frozen identity."""
import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch


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
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def restore(path, model, optimizer, sampler, counters, fingerprint):
    # Only locally generated, trusted checkpoints may be loaded here.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('version') != 4 or state.get('fingerprint') != fingerprint:
        raise ValueError('Checkpoint source/config/data/model fingerprint mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    sampler.load_state_dict(state['sampler'])
    if set(state['counters']) != set(vars(counters)):
        raise ValueError('Checkpoint counter schema mismatch')
    for key, value in state['counters'].items():
        setattr(counters, key, value)
    restore_rng(state['rng'])
