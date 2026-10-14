import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import patr_core as c

CFG=json.loads((c.ROOT/'configs/frozenlake_05b.json').read_text())


def branch(task_id=0,reward=0,n=3):
    return c.Branch(task_id,None,None,[1]*n,[1]*n,[-1.]*n,reward=reward,terminal='success' if reward else 'hole')


def test_task_split_reproducible():
    d=c.make_datasets()
    assert len(d['train'])==10000 and len(d['test'])==100
    assert d['train'][0]['seed']==15795
    assert not {t['uid'] for t in d['train']}&{t['uid'] for t in d['test']}
    assert d==c.make_datasets()


def test_clone_is_independent_and_replays_state():
    task=dict(seed=42,size=4,p=.8)
    env=c.make_env(task,CFG)
    b=branch();b.env=env;b.actions=[3];b.ids=[1,2]
    clone=b.clone('.1')
    clone.ids.append(9);clone.actions.append(1)
    assert b.ids==[1,2] and b.actions==[3]
    clone.env.step(2)
    replay=copy.deepcopy(env)
    replay.step(2)
    assert clone.env.s==replay.s
    assert clone.env is not env


def test_routing_replaces_parents_and_keeps_failed_for_learning(monkeypatch):
    bs=[branch(n=3) for _ in range(4)]
    for i,b in enumerate(bs):b.lineage=str(i);b.reward=[.95,.8,.7,.1][i];b.terminal='';b.turns=5
    monkeypatch.setattr(c,'score_branch',lambda b,cfg:b.reward)
    keep,done,events=c.route_patr(bs,CFG,np.random.default_rng(1))
    assert len(keep)==5 and len(done)==1
    assert done[0].lineage=='3' and done[0].terminal=='pruned'
    assert {x.lineage for x in keep}=={'0.0','0.1','1.0','1.1','2'}
    assert keep[0].ids==bs[0].ids and keep[0].ids is not bs[0].ids


def test_random_expansion_not_biased_by_score_ties(monkeypatch):
    monkeypatch.setattr(c,'score_branch',lambda b,cfg:0.)
    bs=[branch() for _ in range(4)]
    for i,b in enumerate(bs):b.lineage=str(i)
    selected=set()
    for seed in range(20):
        _,_,events=c.route_patr(copy.deepcopy(bs),CFG,np.random.default_rng(seed),True)
        selected|={x['lineage'] for x in events if x['kind']=='expand'}
    assert selected=={'0','1','2','3'}


def test_group_weighting_and_zero_variance():
    bs=[branch(0,0,2),branch(0,1,5),branch(1,0,3),branch(1,0,4),branch(1,0,5)]
    rows=c.training_weights(bs,2)
    assert sum(b.ntokens*w for b,a,w in rows)==pytest.approx(1)
    assert sum(b.ntokens*w for b,a,w in rows if b.task_id==0)==pytest.approx(.5)
    assert all(a==0 for b,a,w in rows if b.task_id==1)
    assert rows[0][1]<0<rows[1][1]
    dr=c.training_weights(bs,2,True)
    assert len({w for b,a,w in dr})==1
    assert sum(b.ntokens*w for b,a,w in dr)==pytest.approx(1)


def test_metrics_distinguish_success_from_pass4():
    bs=[branch(t,int(t==0 and k==0)) for t in range(2) for k in range(4)]
    m=c.evaluation_metrics(bs,[{},{}])
    assert m['success_rate_pct']==12.5 and m['pass_at_4_pct']==50


def test_prefix_tokens_and_mask_match_chat_template():
    from transformers import AutoTokenizer
    path=c.ROOT/CFG['model']
    if not path.exists():pytest.skip('requires downloaded tokenizer')
    tok=AutoTokenizer.from_pretrained(path)
    b=c.start_branch(dict(seed=42,size=4,p=.8),0,tok,CFG,'0')
    text='I will check the map. ```invalid```'
    ids=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]
    out=SimpleNamespace(token_ids=ids,logprobs=[{i:SimpleNamespace(logprob=-1.)} for i in ids],text=text,finish_reason='stop')
    c.apply_response(b,out,tok,CFG)
    assert b.ids==tok.apply_chat_template(b.agent.chat_completions,tokenize=True,add_generation_prompt=True)
    assert sum(b.mask)==len(ids) and len(b.ids)==len(b.mask)==len(b.old_logprobs)


def test_process_scorer_does_not_modify_outcome_reward():
    env=SimpleNamespace(s=0,ncol=3,nrow=2,desc=np.array([list('SFG'),list('FHF')],dtype='S1'))
    b=branch(reward=0);b.env=env;b.initial_distance=2
    assert c.distance(env)==2 and c.score_branch(b,CFG)==0
    env.s=1
    assert c.score_branch(b,CFG)==.5
    assert b.reward==0
    env.s=3
    assert c.distance(env)==3 and c.score_branch(b,CFG)==0



@pytest.mark.parametrize('method,expected', [('grpo',16),('dapo',16),('patr',12),('tree_random',12),('arpo',16)])
def test_async_scheduler_preserves_groups_and_is_order_independent(method,expected):
    from transformers import AutoTokenizer
    path=c.ROOT/CFG['model']
    if not path.exists():pytest.skip('requires tokenizer')
    tok=AutoTokenizer.from_pretrained(path)
    text='```invalid```'
    tokens=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]
    class Backend:
        def __init__(self,reverse):self.requests={};self.reverse=reverse
        def add_request(self,rid,prompt,params):self.requests[rid]=(prompt,params)
        def step(self):
            rid=list(self.requests)[-1 if self.reverse else 0]
            self.requests.pop(rid)
            out=SimpleNamespace(token_ids=tokens,logprobs=[{t:SimpleNamespace(logprob=-1.)} for t in tokens],text=text,finish_reason='stop')
            return [SimpleNamespace(request_id=rid,finished=True,outputs=[out])]
    tasks=c.make_datasets()['train'][:2]
    def run(reverse):
        engine=SimpleNamespace(llm_engine=Backend(reverse))
        bs,stats,events=c.collect(tasks,engine,tok,CFG,method,np.random.default_rng(4))
        assert len(bs)==expected
        assert all(b.turns==10 and b.terminal=='turn_limit' for b in bs)
        assert sum(stats['terminal_counts'].values())==expected
        return [(b.task_id,b.lineage,b.reward,b.ids) for b in bs],stats,sorted(events,key=lambda e:(e['task_id'],e['turn'],str(e.get('lineage',''))))
    assert run(False)==run(True)


def test_valid_action_reaches_goal_and_only_outcome_reward_is_used(monkeypatch):
    from transformers import AutoTokenizer
    path=c.ROOT/CFG['model']
    if not path.exists():pytest.skip('requires tokenizer')
    tok=AutoTokenizer.from_pretrained(path)
    env=c.FrozenLakeEnv(size=2,desc=['SG','FF'],is_slippery=False,max_steps=10)
    env.reset()
    monkeypatch.setattr(c,'make_env',lambda task,cfg:env)
    b=c.start_branch(dict(seed=42,size=2,p=.8),0,tok,CFG,'0')
    text='```Right```';tokens=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]
    out=SimpleNamespace(token_ids=tokens,logprobs=[{t:SimpleNamespace(logprob=-1.)} for t in tokens],text=text,finish_reason='stop')
    c.apply_response(b,out,tok,CFG)
    assert b.actions==[3] and b.reward==1.0 and b.terminal=='success' and b.turns==1
