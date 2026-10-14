import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import dataset_v2 as d
from rllm.environments.frozenlake import frozenlake as f


TEN = ['FFHFFFFF', 'FFFFFHFF', 'FHFFFHFS', 'HFHHHFHF',
       'HFFHHFFF', 'FFFFFFFF', 'FGFFFFFF', 'HFFHHHFH']
ELEVEN = ['HFFHFFFF', 'HFFFFHHF', 'GFFHFFFF', 'FFHFFHHF',
          'FFFFHFHS', 'HFHFHHFF', 'FFFFFHFF', 'FFFFFFFF']


def test_exact_ten_move_budget_and_legacy_off_by_one():
    assert f.shortest_path_length(TEN) == 10
    assert f.shortest_path_length(ELEVEN) == 11
    assert f.is_valid(TEN, 8, 10)
    assert not f.is_valid(TEN, 8, 9)
    assert not f.is_valid(ELEVEN, 8, 10)
    assert f.is_valid(ELEVEN, 8, 11)
    assert f.is_valid(ELEVEN, 8, 10, generation_mode='legacy')


def test_unsolvable_map():
    board = ['SH', 'HG']
    assert f.shortest_path_length(board) is None
    assert not f.is_valid(board, 2, 10)
    assert not f.is_valid(board, 2, 10, generation_mode='legacy')
    with pytest.raises(ValueError, match='within max_steps'):
        f.FrozenLakeEnv(size=2, desc=board, max_steps=10, generation_mode='strict_bfs')
    # Legacy fixed evaluations can still replay impossible/over-budget maps.
    env = f.FrozenLakeEnv(size=8, desc=ELEVEN, max_steps=10, generation_mode='legacy')
    env.reset()
    assert f.shortest_path_length(env.desc) == 11


def test_long_first_dfs_discovery_must_not_block_short_path():
    board = ['FFSF', 'FFFF', 'FFFF', 'FGHF']
    # The old LIFO order marks (1,1) at distance 4, then ignores distance 2;
    # and marks (1,2) at distance 3, then ignores the direct distance 1.
    # A valid route is Left, Down, Down, Down (4 moves).
    assert f.shortest_path_length(board) == 4
    assert not f.is_valid(board, 4, 4, generation_mode='legacy')
    assert f.is_valid(board, 4, 4)
    env = f.FrozenLakeEnv(size=4, desc=board, max_steps=4, generation_mode='strict_bfs')
    env.reset()
    for action in (1, 2, 2, 2):
        _, reward, done, _ = env.step(action)
    assert reward == 1 and done and env.success()


def test_reset_keeps_instance_budget_and_other_environment_unchanged():
    original_global = f.MAX_STEPS
    first = f.FrozenLakeEnv(size=8, seed=72, p=.75, max_steps=10, generation_mode='strict_bfs')
    second = f.FrozenLakeEnv(size=4, seed=19, p=.8, max_steps=2, generation_mode='strict_bfs')
    first_desc, second_desc = first.desc.copy(), second.desc.copy()
    first.reset(); first.step(2)
    second.reset(); first.reset()
    assert first.max_steps == first.env_kwargs['max_steps'] == 10
    assert second.max_steps == second.env_kwargs['max_steps'] == 2
    assert first.generation_mode == 'strict_bfs'
    assert f.MAX_STEPS == original_global
    assert np.array_equal(first.desc, first_desc)
    assert np.array_equal(second.desc, second_desc)
    assert first.desc[first._get_player_position()] == b'S'


def test_legacy_map_snapshot_and_reset_replay():
    desc, goal = f.generate_random_map(size=4, p=.8, seed=42, max_steps=10, generation_mode='legacy')
    assert desc == ['FFSF', 'FHFF', 'FFFH', 'GHFF']
    assert goal == (3, 0)
    env = f.FrozenLakeEnv(size=4, p=.8, seed=42, max_steps=10)
    env.reset()
    assert f.normalize_desc(env.desc) == desc
    sequence = [1, 2, 2, 1, 2]
    replay = [env.step(action)[:3] for action in sequence]
    env.reset()
    assert [env.step(action)[:3] for action in sequence] == replay


def test_optional_compare_unmodified_v1_environment():
    path = d.ROOT.parent / 'patr-reproduction/vendor/rllm_source/rllm/environments/frozenlake/frozenlake.py'
    if not path.exists():
        pytest.skip('v1 checkout unavailable; static legacy snapshot still tested')
    spec = importlib.util.spec_from_file_location('frozenlake_legacy_for_test', path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    for size in (2, 4, 8):
        for seed in (1, 42, 72, 15795):
            for budget in (5, 10):
                old = legacy.FrozenLakeEnv(size=size, seed=seed, p=.75, max_steps=budget)
                new = f.FrozenLakeEnv(size=size, seed=seed, p=.75, max_steps=budget, generation_mode='legacy')
                assert np.array_equal(old.desc, new.desc)
                assert old.reset() == new.reset()
                for action in (0, 1, 2, 3, 4):
                    assert old.step(action) == new.step(action)


def test_from_dict_preserves_explicit_desc_and_budget():
    task = dict(size=2, seed=999, p=.6, desc=['SG', 'HH'], max_steps=10,
                generation_mode='strict_bfs', is_slippery=False)
    env = f.FrozenLakeEnv.from_dict(task)
    env.reset()
    assert f.normalize_desc(env.desc) == task['desc']
    assert env.max_steps == 10
    assert env.step(3)[1:3] == (1, True)


@pytest.fixture(scope='module')
def small_dataset():
    return d.build_dataset(seed=20261014, train=64, validation=16, test=16, holdout=16)


def test_dataset_reproducibility_disjoint_geometry_and_budgets(small_dataset):
    data, manifest = small_dataset
    assert (data, manifest) == d.build_dataset(seed=20261014, train=64, validation=16, test=16, holdout=16)
    report = d.validate_dataset(data, 10)
    assert sum(row['rows'] for row in report.values()) == 112
    assert all(t['shortest_distance'] <= 10 for rows in data.values() for t in rows)
    for split in ('validation', 'test', 'holdout'):
        assert report[split]['duplicate_geometry_rows'] == 0
    for split, rows in data.items():
        assert len(set(t['size'] for t in rows)) == 8
        counts = manifest['rejection_counts'][split]
        assert counts['map_draws'] == len(rows) + counts['cross_split_geometry'] + counts['within_eval_geometry']
        assert counts['board_candidates'] == counts['map_draws'] + counts['no_path'] + counts['over_budget']
        for task in rows:
            generated, _ = f.generate_random_map(size=task['size'], p=task['p'], seed=task['seed'],
                max_steps=task['max_steps'], generation_mode='strict_bfs')
            assert generated == task['desc']


def test_task_manifest_detects_corruption(small_dataset):
    task = copy.deepcopy(small_dataset[0]['train'][0])
    task['shortest_distance'] += 1
    with pytest.raises(ValueError, match='metadata'):
        d.verify_task(task)
    task = copy.deepcopy(small_dataset[0]['train'][0])
    task['geometry_hash'] = 'wrong'
    with pytest.raises(ValueError, match='hash'):
        d.verify_task(task)
    with pytest.raises(ValueError, match='budget'):
        d.verify_task(small_dataset[0]['train'][0], max_steps=0)


def test_dataset_rejects_cross_split_overlap(small_dataset):
    data = copy.deepcopy(small_dataset[0])
    data['test'][0] = {**data['train'][0], 'split': 'test', 'uid': 'different_uid'}
    with pytest.raises(ValueError, match='multiple splits'):
        d.validate_dataset(data)


def test_save_is_versioned_and_does_not_replace_old_data(tmp_path, small_dataset):
    data, manifest = small_dataset
    dest = tmp_path / 'data_v2.json'
    saved = d.save_dataset(dest, data, manifest)
    assert saved['data_sha256'] == hashlib.sha256(dest.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        d.save_dataset(dest, data, manifest)
    with pytest.raises(FileExistsError):
        d.save_dataset(tmp_path / 'data.json', data, manifest)


def test_finite_2x2_space_cannot_hang():
    with pytest.raises(ValueError, match='finite geometry pool'):
        d.build_dataset(train=10000, validation=200, test=200, holdout=200)


@pytest.mark.parametrize('desc', [['SS', 'FG'], ['SH', 'FF'], ['SX', 'FG'], ['SFF', 'FG']])
def test_malformed_explicit_maps_fail(desc):
    with pytest.raises(ValueError):
        f.normalize_desc(desc)
