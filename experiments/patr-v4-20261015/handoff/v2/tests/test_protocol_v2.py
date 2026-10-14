import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

import patr_core as c
from protocol_v2 import CorrectedFrozenLakeAgent, FORMAT_REMINDER, profile_config

BASE = json.loads((c.ROOT / 'configs/frozenlake_05b.json').read_text())


class Tokenizer:
    eos_token_id = 1
    def __len__(self): return 500
    def get_vocab(self): return {str(i):i for i in range(500)}
    def encode(self, text, **kwargs): return [ord(ch)+2 for ch in text]
    def apply_chat_template(self, messages, **kwargs): return self.encode(json.dumps(messages))


def response(text, finish_reason='stop'):
    ids = Tokenizer().encode(text) + [1]
    return SimpleNamespace(token_ids=ids, logprobs=[{t:SimpleNamespace(logprob=-1.)} for t in ids],
                           text=text, finish_reason=finish_reason, stop_reason=1)


def cfg(profile='corrected_v2'):
    return dict(BASE, **profile_config(profile))


def task():
    return dict(seed=42, size=2, p=.8, desc=['SG','FF'], uid='fixed', generation_mode='legacy')


def start(config=None):
    return c.start_branch(task(), 0, Tokenizer(), config or cfg(), '0')


@pytest.mark.parametrize('text', ['```Up```', '```Down```', '```Left```', '```Right```',
    '```1```', '```4```', 'thought\n``` right ```', 'G', '0', '```Right``', '```G```',
    '```Up``` then ```Down```', '```5```', '```0```'])
def test_parser_stays_exactly_legacy(text):
    legacy = c.FrozenLakeAgent(max_steps=10)
    corrected = CorrectedFrozenLakeAgent(10)
    assert legacy._parse_model_response(text) == corrected._parse_model_response(text)


def test_equal_initial_and_subsequent_format_reminders_and_no_false_invalid():
    b = start()
    assert 'there is no slipping' in b.agent.chat_completions[0]['content']
    assert FORMAT_REMINDER in b.agent.chat_completions[-1]['content']
    c.apply_response(b, response('```Left```'), Tokenizer(), cfg())
    message = b.agent.chat_completions[-1]['content']
    assert FORMAT_REMINDER in message and 'previous action was valid' in message
    assert 'did not contain a valid action' not in message
    assert b.trace[-1]['valid_noop'] and b.trace[-1]['action_valid']
    c.apply_response(b, response('G'), Tokenizer(), cfg())
    assert 'did not contain a valid action' in b.agent.chat_completions[-1]['content']
    assert b.trace[-1]['format_status'] == 'missing_or_incomplete_fence'


def test_behavior_penalty_is_never_an_outcome_or_evaluation_reward():
    config = cfg('behavior_variant_v2')
    train, evaluation = start(config), start(config)
    for b, ev in ((train,False),(evaluation,True)):
        c.apply_response(b, response('G'), Tokenizer(), config, evaluation=ev)
        c.apply_response(b, response('```Right```'), Tokenizer(), config, evaluation=ev)
        assert b.reward == 1 and b.terminal == 'success'
    assert train.training_reward == pytest.approx(.95)
    assert evaluation.training_reward == 1
    metrics = c.evaluation_metrics([copy.deepcopy(train) for _ in range(4)], [task()])
    assert metrics['success_rate_pct'] == metrics['pass_at_4_pct'] == 100


def test_every_turn_repeat_prune_is_reachable_at_six_but_never_in_eval():
    b = start()
    for turn in range(1,7):
        c.apply_response(b,response('G'),Tokenizer(),cfg())
        event = c.prune_repeated_response(b,cfg(),'patr',evaluation=False)
        assert bool(event) == (turn == 6)
    assert b.terminal == 'pruned' and b.turns == 6
    b = start()
    for _ in range(6): c.apply_response(b,response('G'),Tokenizer(),cfg(),evaluation=True)
    assert c.prune_repeated_response(b,cfg(),'patr',evaluation=True) is None
    assert not b.terminal


def test_group_token_weights_keep_tasks_equal_and_common_prefix_cancels():
    def branch(t, r, length):
        return c.Branch(t,None,None,[2]*length,[1]*length,[-1.]*length,reward=r)
    bs = [branch(0,0,20),branch(0,1,2),branch(1,0,3),branch(1,1,9),branch(1,1,8)]
    rows = c.training_weights(bs,2,cfg=cfg('behavior_variant_v2'))
    for t in (0,1):
        group = [(b,a,w) for b,a,w in rows if b.task_id == t]
        assert sum(b.ntokens*w for b,a,w in group) == pytest.approx(.5)
        assert sum(a*w for b,a,w in group) == pytest.approx(0,abs=1e-12)
        assert len({w for b,a,w in group}) == 1
    old = c.training_weights(bs,2,cfg=cfg('legacy'))
    assert sum(a*w for b,a,w in old if b.task_id == 0) > 0
    dapo = c.training_weights(bs,2,True,cfg=cfg('legacy'))
    assert len({w for b,a,w in dapo}) == 1


def test_response_trace_preserves_every_sampled_id_logprob_eos_and_seed():
    b = start()
    prefix = list(b.ids)
    out = response('```Left```')
    c.apply_response(b,out,Tokenizer(),cfg(),sampling_seed=123)
    assert b.ids[:len(prefix)] == prefix
    assert b.ids[len(prefix):len(prefix)+len(out.token_ids)] == out.token_ids
    assert [t for t,m in zip(b.ids,b.mask) if m] == out.token_ids
    assert len(b.ids) == len(b.mask) == len(b.old_logprobs)
    tr = b.trace[0]
    assert tr['response_token_ids'] == out.token_ids and tr['sampling_seed'] == 123
    assert tr['response_logprobs'] == [-1.]*len(out.token_ids)
    assert tr['response_tokens_excluding_eos'] == len(out.token_ids)-1
    assert tr['finish_reason'] == 'stop' and tr['stop_reason'] == 1


def test_evaluation_metrics_report_diversity_turns_lengths_and_distance():
    bs = []
    for first in ('G', '```Left```', '```Right```', '```Down```'):
        b = start()
        c.apply_response(b,response(first),Tokenizer(),cfg(),evaluation=True,sampling_seed=321)
        bs.append(b)
    metrics = c.evaluation_metrics(bs,[task()])
    assert metrics['first_turn_invalid_fraction'] == .25
    assert metrics['valid_noop_fraction'] == .25
    assert metrics['success_rate_pct'] == 25 and metrics['pass_at_4_pct'] == 100
    assert metrics['unique_action_sequences_per_task'] == 4
    assert metrics['identical_four_action_sequences_fraction'] == 0
    assert metrics['response_length']-metrics['response_length_excluding_eos'] == 1
    assert metrics['sampling_seed_missing_count'] == 0
    assert metrics['by_shortest_distance']['1']['success_rate_pct'] == 25
    assert metrics['by_turn']['1']['n'] == 4


def test_random_process_pruning_switch_and_random_patr_ties(monkeypatch):
    monkeypatch.setattr(c,'score_branch',lambda b,config:float(b.lineage)/4)
    bs = [start() for _ in range(4)]
    for i,b in enumerate(bs): b.lineage=str(i)
    conf = cfg(); conf.update(top_k_branches=0, prune_margin=0)
    keep,done,_ = c.route_patr(copy.deepcopy(bs),conf,np.random.default_rng(1),True)
    assert len(keep) == 4 and not done
    conf['tree_random_process_pruning'] = True
    _,done,_ = c.route_patr(copy.deepcopy(bs),conf,np.random.default_rng(1),True)
    assert len(done) == 2
    monkeypatch.setattr(c,'score_branch',lambda b,config:0.)
    conf.update(top_k_branches=1)
    winners = set()
    for seed in range(20):
        _,_,events = c.route_patr(copy.deepcopy(bs),conf,np.random.default_rng(seed))
        winners.update(e['lineage'] for e in events if e['kind']=='expand')
    assert winners == {'0','1','2','3'}


def test_async_scheduler_applies_v2_pruning_and_saves_seed():
    class Backend:
        def __init__(self): self.requests = {}
        def add_request(self,rid,prompt,params): self.requests[rid]=(prompt,params)
        def step(self):
            rid = next(iter(self.requests)); self.requests.pop(rid)
            return [SimpleNamespace(request_id=rid,finished=True,outputs=[response('G')])]
    bs,stats,events = c.collect([task()],SimpleNamespace(llm_engine=Backend()),
                               Tokenizer(),cfg(),'patr',np.random.default_rng(1))
    assert all(b.turns == 6 and b.terminal == 'pruned' for b in bs)
    assert all(t['sampling_seed'] is not None for b in bs for t in b.trace)
    assert any(e['kind']=='prune_repeat' for e in events)
    assert stats['first_turn_invalid_fraction'] == 1


def test_length_terminated_response_is_audited_without_execution():
    b = start()
    c.apply_response(b,response('```Right```','length'),Tokenizer(),cfg(),sampling_seed=17)
    assert b.turns == 0 and b.actions == [] and b.terminal == 'token_limit'
    assert b.trace[0]['turn'] == 1 and b.trace[0]['format_status'] == 'truncated'
    assert b.trace[0]['response_token_ids'] == response('```Right```').token_ids


def test_arpo_vocabulary_contract_covers_evaluation_and_leaves_other_methods_full():
    tokenizer = Tokenizer()
    config = cfg()
    assert c.sampling_allowed_token_ids(tokenizer,config,'arpo') == list(range(500))
    assert c.sampling_allowed_token_ids(tokenizer,config,'grpo') is None
    assert c.sampling_allowed_token_ids(tokenizer,dict(config,active_method='arpo'),'grpo') == list(range(500))
    assert c.sampling_allowed_token_ids(tokenizer,cfg('legacy'),'arpo') is None
    meta = c.sampling_vocabulary_metadata(tokenizer,config,'arpo')
    assert meta['allowed_token_count'] == 500 and meta['entropy_denominator_vocab_size'] == 500
    assert len(meta['allowed_token_ids_sha256']) == 64


def test_arpo_actual_sampling_params_use_same_ids_in_eval():
    seen = []
    class Backend:
        def __init__(self): self.requests = {}
        def add_request(self,rid,prompt,params):
            seen.append(params.allowed_token_ids)
            self.requests[rid]=(prompt,params)
        def step(self):
            rid = next(iter(self.requests)); self.requests.pop(rid)
            return [SimpleNamespace(request_id=rid,finished=True,outputs=[response('```Right```')])]
    config = dict(cfg(),active_method='arpo')
    bs,stats,_ = c.collect([task()],SimpleNamespace(llm_engine=Backend()),Tokenizer(),config,
                          'grpo',np.random.default_rng(4),evaluation=True)
    assert len(bs)==4 and all(ids==list(range(500)) for ids in seen)
    assert stats['sampling_vocabulary']['policy'] == 'tokenizer_vocab'
