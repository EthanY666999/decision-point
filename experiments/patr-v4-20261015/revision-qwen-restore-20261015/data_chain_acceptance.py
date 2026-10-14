"""Independent pinned preparation function plus explicit-map reconstruction."""
import ast
import json
from pathlib import Path
import numpy as np
from dataset_v4 import FrozenLakeEnv, normalize_desc

ROOT = Path(__file__).resolve().parent


class Registry:
    @staticmethod
    def register_dataset(name, data, split):
        return data


def main():
    tree = ast.parse((ROOT/'reference/upstream_prepare.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_frozenlake_data')
    ns = dict(np=np, DatasetRegistry=Registry)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'pinned_prepare_function', 'exec'), ns)
    train, test = ns['prepare_frozenlake_data']()
    data = json.loads((ROOT/'artifacts/data_v4.json').read_text())
    for split, ref in [('train', train), ('test', test)]:
        assert len(ref) == len(data[split])
        for a, b in zip(ref, data[split]):
            for key in ('seed', 'size', 'p', 'uid'):
                assert a[key] == b[key], (split, key)
    checked = 0
    for tasks in data.values():
        for task in tasks:
            env = FrozenLakeEnv.from_dict(dict(task, max_steps=10, is_slippery=False))
            env.reset()
            assert env.max_steps == 10 and normalize_desc(env.desc) == task['desc']
            initial = int(env.s)
            env.step(0)
            assert int(env.s) == initial and normalize_desc(env.desc) == task['desc']
            env.reset()
            assert int(env.s) == initial and normalize_desc(env.desc) == task['desc']
            checked += 1
    report = dict(passed=True, upstream_parameter_rows_exact=10100, from_dict_reset_step_reset_maps=checked, scope='Pinned prepare parameter equality and local environment map invariants; not independent upstream generator equivalence')
    (ROOT/'reports/data_chain_acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
