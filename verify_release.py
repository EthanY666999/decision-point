"""Audit a downloaded FLH Golden Trace release; does not call a model or GPU."""
import argparse
import collections
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

def sha(data):
    return hashlib.sha256(data).hexdigest()

def distance(rows):
    size = len(rows)
    start = next((r, c) for r in range(size) for c in range(size) if rows[r][c] == 'S')
    queue = collections.deque([(start, 0)])
    seen = {start}
    while queue:
        (r, c), d = queue.popleft()
        if rows[r][c] == 'G':
            return d
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            p = r + dr, c + dc
            if 0 <= p[0] < size and 0 <= p[1] < size and rows[p[0]][p[1]] != 'H' and p not in seen:
                seen.add(p)
                queue.append((p, d + 1))
    raise AssertionError('Unreachable map')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('release', nargs='?', default='.')
    parser.add_argument('--expected-count', type=int, default=1000)
    args = parser.parse_args()
    root = Path(args.release).resolve()
    summary = json.loads((root / 'summary.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, meta in summary['files'].items():
        payload = (root / name).read_bytes()
        assert len(payload) == meta['bytes'] and sha(payload) == meta['sha256'], name
    for name, expected in manifest['source_sha256'].items():
        assert sha((root / name).read_bytes()) == expected, name
    blocked = json.loads((root / 'heldout_map_hashes.json').read_text())
    blocked_hashes = set(blocked['map_sha256'])
    blocked_ids = set(blocked['uid'])
    assert len(blocked_hashes) == 700 and len(blocked_ids) == 700
    train_hashes = json.loads((root / 'train_map_hashes.json').read_text())
    sys.path.insert(0, str(root / 'source/rllm_source'))
    from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
    from rllm.agents.frozenlake_agent import FrozenLakeAgent
    with gzip.open(root / 'golden_traces.jsonl.gz', 'rt') as f:
        records = [json.loads(line) for line in f if line.strip()]
    with gzip.open(root / 'sft_messages.jsonl.gz', 'rt') as f:
        sft = [json.loads(line) for line in f if line.strip()]
    teachers = {}
    with gzip.open(root / 'teacher_calls.jsonl.gz', 'rt') as f:
        for line in f:
            call = json.loads(line)
            assert call['requested_model'] == 'gpt-6.1-sol' and call['fallback_model'] is None and call['returncode'] == 0
            games = {g['id']: g['rows'] for g in call['request']['games']}
            answers = json.loads(call['response_text'])['answers']
            assert len(answers) == len(games) and {a['id'] for a in answers} == set(games)
            for answer in answers:
                uid = answer['id']
                assert uid not in teachers
                teachers[uid] = dict(batch=call['request']['batch_id'], rows=games[uid], actions=answer['actions'])
    assert len(records) == len(sft) == args.expected_count
    assert len({r['task']['uid'] for r in records}) == args.expected_count
    assert len({r['map_hash'] for r in records}) == args.expected_count
    ids = {'Left': 1, 'Down': 2, 'Right': 3, 'Up': 4}
    deltas = {'Left': (0, -1), 'Down': (1, 0), 'Right': (0, 1), 'Up': (-1, 0)}
    for rec, row in zip(records, sft):
        t, rows = rec['task'], rec['map']
        assert teachers[t['uid']] == dict(batch=rec['teacher_batch'], rows=rows, actions=rec['planned_actions'])
        size = len(rows)
        assert all(len(r) == size for r in rows) and 9 <= size <= 12
        assert rec['accepted'] and rec['terminal'] == 'success' and t['split'] == 'train'
        assert t['uid'] not in blocked_ids and rec['map_hash'] not in blocked_hashes
        assert train_hashes[t['uid']] == rec['map_hash']
        assert sha('\n'.join(rows).encode()) == rec['map_hash'] == t['map_sha256']
        assert 7 <= distance(rows) == rec['shortest_distance'] == t['shortest_path_length'] <= 16
        assert 1 <= len(rec['planned_actions']) == len(rec['steps']) <= 20
        env = FrozenLakeEnv(desc=rows, seed=t['seed'], is_slippery=False, max_steps=20)
        env.reset()
        agent = FrozenLakeAgent(max_steps=20)
        agent.update_from_env(env.render(), 0., False, {})
        rr, cc = divmod(int(env.s), size)
        assert rows[rr][cc] == 'S'
        for index, (word, step) in enumerate(zip(rec['planned_actions'], rec['steps'])):
            assert step['turn'] == index + 1 and step['messages_before'] == agent.chat_completions
            assert step['state_before'] == int(env.s) == rr * size + cc
            assert step['response'] == '```' + word + '```'
            act = int(agent.update_from_model(step['response']).action)
            assert act == step['action'] == ids[word]
            dr, dc = deltas[word]
            rr, cc = max(0, min(size-1, rr+dr)), max(0, min(size-1, cc+dc))
            assert rows[rr][cc] != 'H'
            obs, reward, done, info = env.step(act)
            assert int(env.s) == step['state_after'] == rr * size + cc
            assert obs == step['observation_after'] and float(reward) == step['reward']
            assert done == (index == len(rec['steps']) - 1)
            if not done:
                agent.update_from_env(obs, reward, done, info)
        assert env.success() and rows[rr][cc] == 'G' and rec['messages'] == agent.chat_completions
        assert row == dict(task_uid=t['uid'], map_hash=rec['map_hash'], messages=rec['messages'])
    print(json.dumps(dict(verified=len(records), unique_train_maps=len(records),
                         heldout_overlap=0, geometry_and_environment_agree=True)))

if __name__ == '__main__':
    main()
