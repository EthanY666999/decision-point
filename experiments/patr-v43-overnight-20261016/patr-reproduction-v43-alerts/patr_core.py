from eos_metrics import effective_eos_ids, response_metrics
"""Paper-based FrozenLake rollout policies. Original rLLM environment/agent are vendored.

No oracle selects an action: BFS is used only by PaTR's process scorer.
All actions, including those on failures and pruned branches, come from the policy.
"""
import copy
from trajectory_budget import TrajectoryBudget
from tokenization_v4 import encode_message
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor/rllm_source"))
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.agents.frozenlake_agent import FrozenLakeAgent
from protocol_v2 import CorrectedFrozenLakeAgent, format_status


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def append_json(path, value):
    with open(path, "a") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()


def make_datasets(cfg=None):
    """Replicate upstream generation order and seed, without its DatasetRegistry dependency."""
    dest = ROOT / (cfg or {}).get("dataset_path", "artifacts/data.json")
    if dest.exists():
        return json.loads(dest.read_text())
    if (cfg or {}).get("dataset_path"):
        raise FileNotFoundError(f"Explicit dataset must already exist: {dest}")
    rng = np.random.RandomState(42)
    nt, ne = 10000, 100
    ts, es = rng.randint(0, 100000, nt), rng.randint(0, 100000, ne)
    tz, ez = rng.randint(2, 10, nt), rng.randint(2, 10, ne)
    tp, ep = rng.uniform(.6, .85, nt), rng.uniform(.6, .85, ne)
    def rows(seeds, sizes, ps, split):
        return [dict(seed=int(s), size=int(z), p=float(p), uid=f"{s}_{z}_{p}", split=split)
                for s, z, p in zip(seeds, sizes, ps)]
    data = dict(train=rows(ts, tz, tp, "train"), test=rows(es, ez, ep, "test"))
    # Validation is generated separately; the fixed test split is never used for selection.
    vr = np.random.RandomState(20261013)
    data["validation"] = rows(vr.randint(100000, 200000, 100), vr.randint(2, 10, 100),
                              vr.uniform(.6, .85, 100), "validation")
    assert not ({x['uid'] for x in data['train']} & {x['uid'] for x in data['test']})
    atomic_json(dest, data)
    atomic_json(ROOT / "artifacts/data_manifest.json", {
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "sizes": {k:len(v) for k,v in data.items()}, "seed":42,
        "source": "rLLM 7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4",
        "p_means": "probability of a frozen tile, NOT slip probability",
        "author_split_verified": False,
        "note": "Task UIDs disjoint; small maps can still have duplicate geometry. See map_audit.json."
    })
    return data


def make_env(task, cfg):
    if not task.get('desc') or task.get('generation_max_steps') != 8:
        raise ValueError('V4 requires frozen generation-8 maps')
    generation_mode = task.get('generation_mode', cfg.get('map_generation_mode', 'legacy'))
    if generation_mode == 'strict_bfs' and task.get('desc') is not None:
        from dataset_v2 import verify_task
        verify_task(task, max_steps=cfg['max_turns'])
    e = FrozenLakeEnv(size=task["size"], p=task["p"], seed=task["seed"],
                      is_slippery=cfg["is_slippery"], max_steps=cfg["max_turns"],
                      desc=task.get('desc'), generation_mode=generation_mode)
    e.reset()
    return e


def distance(env):
    start = int(env.s)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        pos, d = queue.popleft()
        r, c = divmod(pos, env.ncol)
        if env.desc[r, c] == b"G":
            return d
        for dr, dc in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            rr, cc = r + dr, c + dc
            nxt = rr * env.ncol + cc
            if 0 <= rr < env.nrow and 0 <= cc < env.ncol and nxt not in seen and env.desc[rr, cc] != b"H":
                seen.add(nxt)
                queue.append((nxt, d+1))
    return None


@dataclass
class Branch:
    task_id: int
    env: object
    agent: object
    ids: list
    mask: list
    old_logprobs: list
    actions: list = field(default_factory=list)
    turns: int = 0
    reward: float = 0.0
    terminal: str = ""
    initial_distance: int = 1
    entropy: float = 0.0
    initial_entropy: object = None
    trace: list = field(default_factory=list)
    lineage: str = ""
    order: tuple = ()
    training_reward: object = None
    budget: object = None

    @property
    def ntokens(self):
        return sum(self.mask)

    def clone(self, suffix, reset_entropy=False):
        child = copy.deepcopy(self)
        child.lineage += suffix
        if reset_entropy:
            child.initial_entropy = None
        return child


def start_branch(task, task_id, tokenizer, cfg, lineage):
    env = make_env(task, cfg)
    profile = cfg.get('prompt_profile', 'legacy')
    if profile not in ('legacy', 'corrected_v2'):
        raise ValueError(f'Unknown prompt profile: {profile}')
    agent = (CorrectedFrozenLakeAgent(cfg['max_turns'], cfg['is_slippery'])
             if profile == 'corrected_v2' else FrozenLakeAgent(max_steps=cfg['max_turns']))
    agent.update_from_env(env.render(), 0., False, {})
    ids = tokenizer.apply_chat_template(agent.chat_completions, tokenize=True, add_generation_prompt=True)
    return Branch(task_id, env, agent, ids, [0]*len(ids), [0.]*len(ids),
                  initial_distance=max(1, distance(env) or 1), lineage=lineage, training_reward=0.,
                  budget=TrajectoryBudget(cfg["max_response_tokens"]))


def score_branch(b, cfg):
    """Explicit reconstruction of the unspecified FrozenLake scoring blend.

    Accumulated reward and clipped progress relative to starting BFS distance,
    combined using frozen, predeclared weights. Never optimized on the test split.
    """
    d = distance(b.env)
    normalization = cfg.get('scorer_distance_norm', 'initial')
    if normalization not in ('initial', 'map_max'):
        raise ValueError('Unknown scorer normalization')
    denominator = b.initial_distance if normalization == 'initial' else b.env.nrow+b.env.ncol-2
    progress = 0. if d is None else max(0., min(1., 1.-d/max(1,denominator)))
    return float(np.clip(cfg['scorer_reward_weight']*b.reward +
                         cfg['scorer_progress_weight']*progress, 0., 1.))


def route_patr(active, cfg, rng, random_selection=False):
    if not active:
        return [], [], []
    scores = np.array([score_branch(b,cfg) for b in active])
    # Uniform expansion control retains the same conservative pruning rule.
    # In particular, ties in zero process scores must not defeat random selection.
    if random_selection:
        order = rng.permutation(len(active))
    elif cfg.get('patr_tie_break', 'stable') == 'random':
        permutation = rng.permutation(len(active))
        order = permutation[np.argsort(-scores[permutation], kind='stable')]
    else:
        order = np.argsort(-scores, kind='stable')
    expand = set(order[:min(cfg['top_k_branches'],len(active))].tolist())
    median = float(np.median(scores))
    keep, done, events = [], [], []
    for i,b in enumerate(active):
        # Match Algorithm 1: expansion selection precedes pruning on A \ E.
        if i in expand:
            for j in range(cfg['branch_factor']):
                child=b.clone(f".{j}");child.order=b.order+(j,)
                keep.append(child)
            events.append(dict(kind="expand", lineage=b.lineage, score=float(scores[i]), turn=b.turns))
        elif ((not random_selection or cfg.get('tree_random_process_pruning', True)) and
              scores[i] < median-cfg['prune_margin'] or repeat_limit_reached(b, cfg)):
            b.terminal = "pruned"
            done.append(b)
            events.append(dict(kind="prune", lineage=b.lineage, score=float(scores[i]), turn=b.turns))
        else:
            keep.append(b)
    return keep, done, events


def repeat_limit_reached(b, cfg):
    limit = cfg['repeat_limit']
    return limit > 0 and len(b.actions) >= limit and len(set(b.actions[-limit:])) == 1


def prune_repeated_response(b, cfg, method, evaluation):
    """Training-only, before the K-turn expansion barrier; reachable at turn six."""
    if (not evaluation and not b.terminal and method in ('patr', 'tree_random')
            and cfg.get('repeat_check_every_turn', False) and repeat_limit_reached(b, cfg)):
        b.terminal = 'pruned'
        return dict(kind='prune_repeat', task_id=b.task_id, lineage=b.lineage, turn=b.turns)
    return None


def arpo_entropy(logprob_dicts, vocab_size):
    # Faithful to ARPO author's implementation: first 20 token positions,
    # returned top-10 (+sampled token) probabilities, normalized by log(|V|).
    # This is a truncated entropy statistic, not full-vocabulary Shannon entropy.
    values = [float(v.logprob) for d in logprob_dicts[:20] for v in d.values()]
    return -sum(math.exp(v)*v for v in values)/math.log(vocab_size)


def sampling_allowed_token_ids(tokenizer, cfg, method=None):
    """Shared sampler/trainer vocabulary contract; never change only one side."""
    policy = cfg.get('arpo_vocabulary_policy', 'full_model_vocab')
    if policy not in ('full_model_vocab', 'tokenizer_vocab'):
        raise ValueError(f'Unknown ARPO vocabulary policy: {policy}')
    is_arpo = method == 'arpo' or cfg.get('active_method') == 'arpo'
    if not is_arpo or policy == 'full_model_vocab':
        return None
    ids = sorted(set(tokenizer.get_vocab().values()))
    if not ids or ids[0] < 0 or tokenizer.eos_token_id not in ids:
        raise ValueError('Tokenizer vocabulary must contain valid nonnegative IDs and EOS')
    return ids


def sampling_vocabulary_metadata(tokenizer, cfg, method=None):
    ids = sampling_allowed_token_ids(tokenizer, cfg, method)
    return dict(policy='tokenizer_vocab' if ids is not None else 'full_model_vocab',
                allowed_token_count=len(ids) if ids is not None else None,
                allowed_token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest() if ids is not None else None,
                entropy_denominator_vocab_size=len(ids) if ids is not None else len(tokenizer))


def apply_response(b, out, tokenizer, cfg, evaluation=False, sampling_seed=None):
    # Upstream trains on canonical re-tokenized messages, not sampled IDs.
    tokens = list(out.token_ids)
    if not tokens or len(tokens) != len(out.logprobs):
        raise ValueError('Empty or incomplete sampler output')
    sampled_lp = [float(lp[t].logprob) for t, lp in zip(tokens, out.logprobs)]
    if not all(math.isfinite(x) for x in sampled_lp):
        raise ValueError('Nonfinite rollout probability')
    text = out.text
    previous = int(b.env.s)
    action = int(b.agent.update_from_model(text).action)
    b.actions.append(action)
    observation, reward, done, info = b.env.step(action)
    b.turns += 1
    b.agent.update_from_env(observation, reward, done, info)
    assistant_message = b.agent.chat_completions[-2]
    environment_message = b.agent.chat_completions[-1]
    assistant, amask = encode_message(assistant_message, tokenizer)
    environment, emask = encode_message(environment_message, tokenizer, generation=True)
    before = len(b.budget.tokens)
    effective_reward, terminal, truncated = b.budget.append_turn(
        assistant, amask, environment, emask, float(reward), done)
    appended = b.budget.tokens[before:]
    b.ids.extend(appended)
    b.mask.extend(b.budget.mask[before:])
    # No sampled probability exists for canonical template tokens. Training must
    # recompute anchors; NaNs prevent accidental reuse as an importance ratio.
    b.old_logprobs.extend([float('nan')] * len(appended))
    b.reward += effective_reward
    b.training_reward = b.reward
    if truncated:
        b.terminal = 'token_limit'
    elif terminal:
        b.terminal = 'success' if b.env.success() else 'hole'
    elif b.turns >= cfg['max_turns']:
        b.terminal = 'turn_limit'
    b.trace.append(dict(turn=b.turns, response=text, response_token_ids=tokens,
        response_logprobs=sampled_lp, response_tokens=len(tokens),
        **response_metrics(tokens, effective_eos_ids(tokenizer, cfg)),
        canonical_assistant_tokens=len(assistant), canonical_environment_tokens=len(environment),
        post_prompt_tokens=b.budget.counted_tokens, stored_post_prompt_tokens=len(b.budget.tokens),
        canonical_differs_from_sampled=assistant != tokens,
        finish_reason=out.finish_reason, sampling_seed=sampling_seed, action=action,
        reward=float(reward), training_reward=effective_reward,
        state_before=previous, state=int(b.env.s), observation=observation,
        format_status=format_status(text, action), action_valid=action != 0,
        position_unchanged=int(b.env.s) == previous,
        valid_noop=action != 0 and int(b.env.s) == previous, terminal=b.terminal))


def collect_batched(tasks, engine, tokenizer, cfg, method, rng, evaluation=False):
    if cfg.get('protocol_profile', 'legacy') != 'legacy':
        raise ValueError('collect_batched is retained only for legacy regression; use collect for v2')
    from vllm import SamplingParams
    allowed_ids = sampling_allowed_token_ids(tokenizer, cfg, method)
    vocabulary_kwargs = dict(allowed_token_ids=allowed_ids) if allowed_ids is not None else {}
    if allowed_ids is not None:
        cfg = dict(cfg, arpo_entropy_vocab_size=len(allowed_ids))
    initial = cfg['eval_samples'] if evaluation else (cfg['flat_group_size'] if method in ('grpo','dapo') else cfg['initial_branches'])
    active, completed, counts, events = [], [], Counter(), []
    for ti,task in enumerate(tasks):
        for j in range(initial):
            active.append(start_branch(task,ti,tokenizer,cfg,str(j)))
        counts[ti] = initial
    generated = input_tokens = 0
    while active:
        viable = []
        for b in active:
            if len(b.ids)>=cfg['max_context_tokens'] or b.ntokens>=cfg['max_response_tokens']:
                b.terminal="context_limit" if len(b.ids)>=cfg['max_context_tokens'] else "token_limit"
                completed.append(b)
            else:
                viable.append(b)
        active = viable
        if not active:
            if method=='arpo' and not evaluation:
                for ti,task in enumerate(tasks):
                    if counts[ti]<cfg['flat_group_size']:
                        active.append(start_branch(task,ti,tokenizer,cfg,f"r{counts[ti]}"))
                        counts[ti]+=1
            if not active:
                break
        params=[]
        for b in active:
            params.append(SamplingParams(
                temperature=cfg['eval_temperature'] if evaluation else cfg['temperature'],
                top_p=cfg['eval_top_p'] if evaluation else 1.,
                top_k=cfg['eval_top_k'] if evaluation else -1,
                max_tokens=min(b.budget.remaining,cfg['max_context_tokens']-len(b.ids)),
                logprobs=10 if method=='arpo' and not evaluation else 1,
                seed=int(rng.integers(0,2**31-1)),
                stop_token_ids=[tokenizer.eos_token_id],
                **vocabulary_kwargs,
            ))
        input_tokens += sum(len(b.ids) for b in active)
        progress_path=os.environ.get('PATR_PROGRESS_PATH')
        if progress_path:
            atomic_json(progress_path,dict(phase='evaluation' if evaluation else 'rollout',method=method,
                 active=len(active),completed=len(completed),generated_tokens=generated,
                 min_turn=min(b.turns for b in active),max_turn=max(b.turns for b in active),updated_unix=time.time()))
        outputs=engine.generate([{'prompt_token_ids':b.ids} for b in active],params,use_tqdm=False)
        assert len(outputs)==len(active)
        survivors=[]
        for b,result in zip(active,outputs):
            out=result.outputs[0]
            generated += len(out.token_ids)
            apply_response(b,out,tokenizer,cfg,evaluation=evaluation)
            if b.terminal:
                completed.append(b)
            else:
                survivors.append(b)
        active=survivors
        if evaluation:
            continue
        if method in ('patr','tree_random'):
            next_active=[]
            for ti in range(len(tasks)):
                bs=[b for b in active if b.task_id==ti]
                if bs and bs[0].turns % cfg['branch_interval']==0:
                    keep,done,ev=route_patr(bs,cfg,rng,method=='tree_random')
                    next_active.extend(keep);completed.extend(done)
                    events.extend(dict(task_id=ti,**x) for x in ev)
                else:
                    next_active.extend(bs)
            active=next_active
        elif method=='arpo':
            extra=[]
            for b in active:
                if counts[b.task_id]>=cfg['flat_group_size']:
                    continue
                delta=b.entropy-b.initial_entropy
                decision=float(np.clip(rng.random()-cfg['arpo_entropy_weight']*delta,0.,1.))
                if decision<=cfg['arpo_branch_probability']:
                    extra.append(b.clone(f".a{counts[b.task_id]}",reset_entropy=True))
                    counts[b.task_id]+=1
                    events.append(dict(kind='arpo_expand',task_id=b.task_id,turn=b.turns,entropy_delta=delta))
            active.extend(extra)
            # Fill unused branch budget by fresh trajectories when all live paths terminate.
            living={b.task_id for b in active}
            for ti,task in enumerate(tasks):
                if ti not in living and counts[ti]<cfg['flat_group_size']:
                    active.append(start_branch(task,ti,tokenizer,cfg,f"r{counts[ti]}"))
                    counts[ti]+=1
    assert all(b.ntokens>0 for b in completed)
    stats=dict(generated_tokens=generated,input_tokens=input_tokens,episodes=len(completed),
               success_rate=float(np.mean([b.reward>0 for b in completed])),
               mean_response_tokens=float(np.mean([b.ntokens for b in completed])),
               mean_turns=float(np.mean([b.turns for b in completed])),
               terminal_counts=dict(Counter(b.terminal for b in completed)),
               branch_events=len(events))
    return completed,stats,events


def evaluation_metrics(branches, tasks):
    group={i:[b for b in branches if b.task_id==i] for i in range(len(tasks))}
    assert all(len(bs)==4 for bs in group.values())
    success=np.array([[float(b.reward>0) for b in bs] for bs in group.values()])
    actions=[a for b in branches for a in b.actions]
    metrics = dict(success_rate_pct=100*float(success.mean()),pass_at_4_pct=100*float(success.max(1).mean()),
                response_length=float(np.mean([b.ntokens for b in branches])),
                turns=float(np.mean([b.turns for b in branches])),n_tasks=len(tasks),n_trajectories=len(branches),
                invalid_action_fraction=sum(a==0 for a in actions)/len(actions) if actions else None,
                terminal_counts=dict(Counter(b.terminal for b in branches)))
    traces = [t for b in branches for t in b.trace]
    first = [b.trace[0] for b in branches if b.trace]
    sampled = [t for t in traces if 'response_tokens' in t]
    executed = [t for t in traces if t.get('action') is not None]
    average = lambda values: float(np.mean(values)) if values else None
    metrics.update(
        first_turn_invalid_fraction=average([not t.get('action_valid', t.get('action') in (1,2,3,4)) for t in first]),
        format_invalid_fraction=average([t.get('format_status') != 'valid' for t in traces if 'format_status' in t]),
        first_turn_format_counts=dict(Counter(t.get('format_status', 'unknown') for t in first)),
        valid_noop_fraction=average([bool(t.get('valid_noop', False)) for t in executed]),
        response_length_excluding_eos=(sum(t['response_tokens_excluding_eos'] for t in sampled)/len(branches) if sampled else None),
        response_tokens_per_turn_including_eos=average([t['response_tokens'] for t in sampled]),
        response_tokens_per_turn_excluding_eos=average([t['response_tokens_excluding_eos'] for t in sampled]),
        sampling_seed_missing_count=sum(t.get('sampling_seed') is None for t in traces),
        unique_action_sequences_per_task=average([len({tuple(b.actions) for b in bs}) for bs in group.values()]),
        identical_four_action_sequences_fraction=average([len({tuple(b.actions) for b in bs}) == 1 for bs in group.values()]),
        unique_response_sequences_per_task=average([len({tuple(t.get('response', '') for t in b.trace) for b in bs}) for bs in group.values()]),
        metric_definition=dict(response_length='mean total sampled assistant token IDs per trajectory, including EOS',
                               success='raw environment outcome > 0, without training penalties',
                               pass_at_4='fraction of tasks with at least one success in four samples'))
    by_turn = {}
    for turn in sorted({t['turn'] for t in traces}):
        ts = [t for t in traces if t['turn'] == turn]
        by_turn[str(turn)] = dict(n=len(ts),
            invalid_fraction=average([not t.get('action_valid', t.get('action') in (1,2,3,4)) for t in ts]),
            response_tokens=average([t['response_tokens'] for t in ts if 'response_tokens' in t]),
            response_tokens_excluding_eos=average([t['response_tokens_excluding_eos'] for t in ts if 'response_tokens_excluding_eos' in t]))
    metrics['by_turn'] = by_turn
    by_distance = {}
    for d in sorted({b.initial_distance for b in branches}):
        selected = [b for b in branches if b.initial_distance == d]
        task_ids = sorted({b.task_id for b in selected})
        by_distance[str(d)] = dict(n_tasks=len(task_ids), n_trajectories=len(selected),
            success_rate_pct=100*float(np.mean([b.reward > 0 for b in selected])),
            pass_at_4_pct=100*float(np.mean([any(b.reward > 0 for b in group[i]) for i in task_ids])),
            response_length=average([b.ntokens for b in selected]), turns=average([b.turns for b in selected]))
    metrics['by_shortest_distance'] = by_distance
    metrics['canonical_assistant_length'] = float(np.mean([b.ntokens for b in branches]))
    metrics['response_length'] = sum(t['response_tokens'] for t in sampled)/len(branches)
    metrics['post_prompt_tokens'] = float(np.mean([b.budget.counted_tokens for b in branches]))
    metrics['stored_post_prompt_tokens'] = float(np.mean([len(b.budget.tokens) for b in branches]))
    metrics['truncation_fraction'] = float(np.mean([b.terminal == 'token_limit' for b in branches]))
    return metrics


def training_weights(branches, n_tasks, dapo=False, cfg=None):
    """Normalize by question, then by tree size and response length (Eq 2/9).

    DAPO instead uses token-normalized PG loss. Zero-variance groups yield zero PG.
    """
    cfg = cfg or {}
    normalization = cfg.get('loss_normalization', 'global_token' if dapo else 'trajectory')
    if dapo:
        normalization = cfg.get('dapo_loss_normalization', normalization)
    if normalization not in ('trajectory', 'group_token', 'global_token'):
        raise ValueError(f'Unknown loss normalization: {normalization}')
    group={i:[b for b in branches if b.task_id==i] for i in range(n_tasks)}
    adv={}
    for bs in group.values():
        assert bs
        rewards=np.array([b.reward if b.training_reward is None else b.training_reward for b in bs],dtype=np.float64)
        std=rewards.std(ddof=1) if len(bs)>1 else 0.
        for b,r in zip(bs,rewards):adv[id(b)]=float((r-rewards.mean())/(std+1e-6))
    ntotal=sum(b.ntokens for b in branches)
    group_tokens = {i: sum(b.ntokens for b in bs) for i,bs in group.items()}
    def weight(b):
        if normalization == 'global_token':
            return 1./ntotal
        if normalization == 'group_token':
            return 1./(n_tasks*group_tokens[b.task_id])
        return 1./(n_tasks*len(group[b.task_id])*b.ntokens)
    return [(b,adv[id(b)],weight(b)) for b in branches]


def collect(tasks, engine, tokenizer, cfg, method, rng, evaluation=False):
    """Stream completed turns back to vLLM without a global all-task barrier.

    PaTR retains a per-task barrier at every K turns. ARPO retains its per-task
    per-turn allocation barrier. Seeds and branch ordering do not depend on
    cross-task completion order. The optimizer still waits for the whole batch.
    """
    if method not in ('patr', 'grpo'):
        raise ValueError('V4 only authorizes PaTR and GRPO')
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    allowed_ids = sampling_allowed_token_ids(tokenizer, cfg, method)
    vocabulary_kwargs = dict(allowed_token_ids=allowed_ids) if allowed_ids is not None else {}
    vocabulary_metadata = sampling_vocabulary_metadata(tokenizer, cfg, method)
    if allowed_ids is not None:
        cfg = dict(cfg, arpo_entropy_vocab_size=len(allowed_ids))
    initial=cfg['eval_samples'] if evaluation else (cfg['flat_group_size'] if method in ('grpo','dapo') else cfg['initial_branches'])
    base_seed=int(rng.integers(0,2**31-1))
    def seed_for(*parts):
        raw=json.dumps([base_seed,*parts],sort_keys=True).encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%(2**31-1)
    routing_rng={i:np.random.default_rng(seed_for(t['uid'],'routing')) for i,t in enumerate(tasks)}
    backend=engine.llm_engine
    serial=getattr(engine,'_patr_request_serial',0)
    pending={};request_seeds={};inflight=Counter();waiting={i:[] for i in range(len(tasks))}
    counts=Counter();completed=[];events=[]
    generated=input_tokens=0
    executed_actions=invalid_actions=0

    def enqueue(b):
        nonlocal serial,input_tokens
        remaining=min(b.budget.remaining,cfg['max_context_tokens']-len(b.ids))
        if remaining<=0:
            b.terminal='context_limit' if len(b.ids)>=cfg['max_context_tokens'] else 'token_limit'
            completed.append(b)
            return
        sampling_seed = seed_for(tasks[b.task_id]['uid'],b.lineage,b.turns)
        params=SamplingParams(temperature=cfg['eval_temperature'] if evaluation else cfg['temperature'],
            top_p=cfg['eval_top_p'] if evaluation else 1.,top_k=cfg['eval_top_k'] if evaluation else -1,
            max_tokens=remaining,logprobs=10 if method=='arpo' and not evaluation else 1,
            seed=sampling_seed,
            stop_token_ids=[tokenizer.eos_token_id],output_kind=RequestOutputKind.FINAL_ONLY,
            **vocabulary_kwargs)
        rid=f'patr-stream-{serial}';serial+=1
        backend.add_request(rid,{'prompt_token_ids':list(b.ids)},params)
        request_seeds[rid]=sampling_seed
        pending[rid]=b;inflight[b.task_id]+=1;input_tokens+=len(b.ids)

    def route_ready(ti):
        if inflight[ti] or evaluation or method in ('grpo','dapo'):
            return
        ready=sorted(waiting[ti],key=lambda b:b.order);waiting[ti]=[]
        if ready and method in ('patr','tree_random'):
            ready,done,ev=route_patr(ready,cfg,routing_rng[ti],method=='tree_random')
            completed.extend(done);events.extend(dict(task_id=ti,**x) for x in ev)
        elif ready and method=='arpo':
            extra=[]
            for b in ready:
                if counts[ti]>=cfg['flat_group_size']:break
                delta=b.entropy-b.initial_entropy
                decision=float(np.clip(routing_rng[ti].random()-cfg['arpo_entropy_weight']*delta,0.,1.))
                if decision<=cfg['arpo_branch_probability']:
                    child=b.clone(f'.a{counts[ti]}',reset_entropy=True);child.order=(counts[ti],)
                    extra.append(child);counts[ti]+=1
                    events.append(dict(kind='arpo_expand',task_id=ti,turn=b.turns,entropy_delta=delta))
            ready.extend(extra)
        if not ready and method=='arpo' and counts[ti]<cfg['flat_group_size']:
            b=start_branch(tasks[ti],ti,tokenizer,cfg,f'r{counts[ti]}');b.order=(counts[ti],)
            ready.append(b);counts[ti]+=1
        for b in ready:enqueue(b)

    for ti,task in enumerate(tasks):
        counts[ti]=initial
        for j in range(initial):
            b=start_branch(task,ti,tokenizer,cfg,str(j));b.order=(j,)
            enqueue(b)
    last_progress=0.
    while pending:
        now=time.time()
        if now >= cfg.get("deadline_unix", float("inf"))-900:
            raise TimeoutError("Deadline protection during rollout; use last committed checkpoint")
        progress_path=os.environ.get('PATR_PROGRESS_PATH')
        if progress_path and now-last_progress>=5:
            atomic_json(progress_path,dict(phase='evaluation' if evaluation else 'rollout',method=method,
                active=len(pending),waiting=sum(map(len,waiting.values())),completed=len(completed),
                generated_tokens=generated,updated_unix=now,scheduler='per_task_async'))
            last_progress=now
        results=backend.step()
        affected=set()
        for result in results:
            if not result.finished:continue
            b=pending.pop(result.request_id)
            ti=b.task_id;inflight[ti]-=1;affected.add(ti)
            out=result.outputs[0];generated+=len(out.token_ids)
            previous_turn=b.turns
            apply_response(b,out,tokenizer,cfg,evaluation=evaluation,
                           sampling_seed=request_seeds.pop(result.request_id))
            repeat_event=prune_repeated_response(b,cfg,method,evaluation)
            if repeat_event:
                events.append(repeat_event)
                b.trace[-1]['terminal']=b.terminal
            if b.turns>previous_turn:
                executed_actions+=1;invalid_actions+=int(b.actions[-1]==0)
            if b.terminal:
                completed.append(b)
            elif not evaluation and (method=='arpo' or method in ('patr','tree_random') and b.turns%cfg['branch_interval']==0):
                waiting[ti].append(b)
            else:
                enqueue(b)
        for ti in sorted(affected):route_ready(ti)
    engine._patr_request_serial=serial
    assert not any(waiting.values()),'Unrouted task barrier'
    assert all(b.ntokens>0 for b in completed)
    completed.sort(key=lambda b:(b.task_id,b.order))
    stats=dict(generated_tokens=generated,input_tokens=input_tokens,episodes=len(completed),
               executed_actions=executed_actions,invalid_actions=invalid_actions,
               invalid_action_fraction=invalid_actions/executed_actions if executed_actions else None,
               success_rate=float(np.mean([b.reward>0 for b in completed])),
               mean_response_tokens=float(np.mean([b.ntokens for b in completed])),
               mean_turns=float(np.mean([b.turns for b in completed])),
               terminal_counts=dict(Counter(b.terminal for b in completed)),branch_events=len(events),
               mean_training_reward=float(np.mean([b.reward if b.training_reward is None else b.training_reward for b in completed])),
               first_turn_invalid_fraction=float(np.mean([not b.trace[0].get('action_valid', False) for b in completed if b.trace])),
               arpo_vocabulary_policy=cfg.get('arpo_vocabulary_policy', 'full_model_vocab'),
               sampling_vocabulary=vocabulary_metadata,
               shared_prefix_token_occurrences=sum(sum(t['response_tokens'] for t in b.trace) for b in completed)-generated)
    return completed,stats,events
