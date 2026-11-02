"""Isolated B3 entrypoint; requires a prepared workspace and GPU lease."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', required=True, choices=['patr', 'tree_random'])
    parser.add_argument('--smoke', action='store_true')
    args, _ = parser.parse_known_args()
    # The exclusive GPU lease is acquired by the future serial supervisor.
    fd = int(os.environ['NEXT_GPU_LEASE_FD'])
    os.fstat(fd)
    protocol = json.loads((ROOT / 'artifacts/b3_protocol.json').read_text())
    assert protocol['B2_pass'] and protocol['selected_K'] == 2
    gate = Path(protocol['B2_gate_path'])
    assert hashlib.sha256(gate.read_bytes()).hexdigest() == protocol['B2_gate_sha256']
    assert json.loads(gate.read_text())['pass_gate']
    cfg = json.loads((ROOT / args.config).read_text())
    assert cfg['steps'] == 400 and cfg['eval_every'] == 25
    assert cfg['branch_interval'] == 2 and cfg['leaf_cap'] == 8
    assert cfg['model'] == protocol['sft_model']
    out = ROOT / 'runs' / ('smoke' if args.smoke else 'formal') / f"{args.method}_seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    names = ['train.py', 'patr_core.py', 'run_b3.py', 'scripts/b2_collect.py', 'scripts/branch_budget.py']
    identity = dict(source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names},
                    config=cfg, method=args.method, smoke=args.smoke, protocol=protocol)
    saved = out / 'b3_identity.json'
    if saved.exists():
        assert json.loads(saved.read_text()) == identity, 'Cannot resume modified B3 code/config'
    else:
        saved.write_text(json.dumps(identity, indent=2) + '\n')
    # Import only after workspace and lease preflight. Frozen modules are copied
    # byte-for-byte into this new workspace; no external core is monkey-patched.
    import patr_core
    import train
    from b2_collect import make_collect
    train.collect = make_collect(patr_core, train.collect)
    train.main()


if __name__ == '__main__':
    main()
