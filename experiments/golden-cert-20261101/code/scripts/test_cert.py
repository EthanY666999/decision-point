"""CPU acceptance: tokenizer, consequences, replay, sampling IDs, grouping and selection."""
import copy,json,math,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import patr_core as core
from cert_entropy import token_map,distribution,entropy,entropies,consequence_classes
from cert_select import select,deduplicate
from cert_collect import replay,initial,prepare_nodes,continuations,make_collect

class Backend:
    def __init__(self,tokenizer):self.pending=[];self.tokenizer=tokenizer
    def add_request(self,rid,prompt,params):self.pending.append((rid,prompt,params))
    def abort_request(self,rid):pass
    def step(self):
        pending,self.pending=self.pending,[];out=[]
        for rid,prompt,params in pending:
            word=['Left','Down','Right','Up'][params.seed%4];text='```'+word+'```'
            ids=self.tokenizer.encode(text,add_special_tokens=False)
            # Preserve a deliberately non-round-trippable sampled suffix token.
            lp=[{t:NS(logprob=-.4)} for t in ids]
            out.append(NS(finished=True,request_id=rid,outputs=[NS(token_ids=ids,logprobs=lp,text=text,finish_reason='stop')]))
        return list(reversed(out))

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/cert_g2_uct.json').read_text())
        cls.tokenizer=AutoTokenizer.from_pretrained(cls.cfg['model'],local_files_only=True)
        cls.mapping,cls.examples=token_map(cls.tokenizer)
        cls.tasks=[dict(size=4,p=.8,seed=s,uid=f'test{s}',split='train') for s in (11,22)]
    def env(self,desc,state):
        e=core.FrozenLakeEnv(desc=desc,size=len(desc),is_slippery=False,max_steps=10);e.reset();e.s=state;return e
    def test_token_mapping(self):
        self.assertEqual(len(self.examples),8)
        for text,e in self.examples.items():self.assertIn(e['first'],self.mapping)
        self.assertNotIn(self.tokenizer.encode('banana',add_special_tokens=False)[0],self.mapping)
        out=NS(token_ids=self.tokenizer.encode('???',add_special_tokens=False),logprobs=[])
        p,d=distribution(out,self.tokenizer,self.mapping);self.assertEqual(p[0],1);self.assertTrue(d['format_missing'])
    def test_entropy(self):
        e=self.env(['SFF','FHF','FFG'],0);self.assertEqual(consequence_classes(e)[1],'STAY');self.assertEqual(consequence_classes(e)[4],'STAY')
        e=self.env(['SHF','HFF','FFG'],0);classes=consequence_classes(e);self.assertEqual(classes[2],classes[3]);self.assertEqual(classes[2],'FAIL')
        self.assertEqual(entropy([1,0,0,0,0]),0);self.assertAlmostEqual(entropy([.2]*5),1)
        for _ in range(100):
            p=np.random.default_rng(_).dirichlet([1]*5);a,h,_=entropies(e,dict(enumerate(p)));self.assertTrue(0<=h<=a+1e-12<=1+1e-12)
        old=e.s;_,reward,done,_=e.step(0);self.assertEqual(e.s,old);self.assertEqual(reward,0);self.assertFalse(done)
    def node(self,j,turn,ids,S,H):
        return dict(j=j,turn=turn,branch=NS(ids=ids),state=1,S=S,H_act=1-H,H_cons=H)
    def test_selection_and_dedup(self):
        ns=[self.node(j,1,[5,6],j/4,j/4) for j in range(4)]
        self.assertEqual(len(deduplicate(ns)),1)
        ns=[self.node(0,1,[1],.9,.1),self.node(1,2,[2],.2,.8),self.node(2,3,[3],.5,.5)]
        self.assertEqual(select(ns,'S',1,.5,42,'u',1)[0]['j'],0)
        self.assertEqual(select(ns,'H_act',1,.5,42,'u',1)[0]['j'],0)
        self.assertEqual(select(ns,'H_cons',1,.5,42,'u',1)[0]['j'],1)
        self.assertEqual(select(ns,'uct',1,0,42,'u',1),select(ns,'S',1,0,42,'u',1))
        self.assertEqual(select(ns,'uct',1,1,42,'u',1),select(ns,'H_cons',1,1,42,'u',1))
        for mode in ('uniform','S','H_act','H_cons','uct'):
            self.assertEqual(select(ns,mode,2,.5,42,'u',1),select(ns,mode,2,.5,42,'u',1))
            self.assertEqual(len(select(ns[:1],mode,2,.5,42,'u',1)),1)
    def test_full_mock_rollout_replay_and_group(self):
        engine=NS(llm_engine=Backend(self.tokenizer))
        first,nodes,cost,_=initial(core,self.tasks,engine,self.tokenizer,self.cfg,123)
        self.assertEqual(len(first),8)
        candidates,ph=prepare_nodes(nodes,first,self.tasks)
        picked={ti:select(ns,'uct',2,ph[ti],123,self.tasks[ti]['uid'],1) for ti,ns in candidates.items()}
        for ti,ns in candidates.items():
            self.assertEqual(sum(n['turn']==1 for n in ns),1)
            for n in ns:
                b=replay(core,n,self.tasks[ti],self.cfg)
                self.assertEqual(b.turns,n['turn']-1);self.assertEqual(10-b.turns,10-n['turn']+1)
                self.assertEqual(b.ntokens,n['branch'].ntokens)
                self.assertEqual(b.old_logprobs,n['branch'].old_logprobs)
        added,_,skips=continuations(core,picked,self.tasks,engine,self.tokenizer,self.cfg,123)
        self.assertFalse(skips)
        leaves=first+[x[0] for x in added.values()]
        self.assertEqual(len(leaves),12)
        self.assertEqual(len(core.training_weights(leaves,2)),12)
        again,_,_=continuations(core,picked,self.tasks,engine,self.tokenizer,self.cfg,123)
        for key,(b,_,_) in added.items():
            self.assertEqual(b.ids,again[key][0].ids);self.assertEqual(b.old_logprobs,again[key][0].old_logprobs)
        n=copy.deepcopy(next(iter(picked.values()))[0]);n['state']+=1
        with self.assertRaises(AssertionError):replay(core,n,self.tasks[0],self.cfg)
    def test_sample_ids_not_retokenized(self):
        b=core.start_branch(self.tasks[0],0,self.tokenizer,self.cfg,'0');boundary=len(b.ids)
        ids=self.tokenizer.encode('```Left```',add_special_tokens=False)+[self.tokenizer.eos_token_id]
        # text omits EOS while sampled IDs include it; re-encoding would lose it.
        out=NS(token_ids=ids,logprobs=[{t:NS(logprob=-.7)} for t in ids],text='```Left```',finish_reason='stop')
        core.apply_response(b,out,self.tokenizer,self.cfg)
        self.assertEqual(b.ids[boundary:boundary+len(ids)],ids)
        self.assertEqual(b.mask[boundary:boundary+len(ids)],[1]*len(ids))
        self.assertEqual(b.old_logprobs[boundary:boundary+len(ids)],[-.7]*len(ids))

    def test_gate_rejects_each_failed_requirement(self):
        from g1_gate import decision
        base={'K2_patr':{'mixed':100,'tokens':1000},'K2_tree_random':{'mixed':101,'tokens':1100}}
        self.assertTrue(all(decision(102,99,.01,1100,base).values()))
        for args in ((102,102,.01,1100),(102,99,0,1100),(100,99,.01,1100),(102,99,.01,1151)):
            self.assertFalse(all(decision(*args,base).values()))
    def test_adapter_statistics_and_evaluation_delegation(self):
        engine=NS(llm_engine=Backend(self.tokenizer));calls=[]
        def original(*args,**kwargs):calls.append(kwargs);return 'evaluation sentinel'
        adapter=make_collect(core,original)
        self.assertEqual(adapter(self.tasks,engine,self.tokenizer,self.cfg,'patr',np.random.default_rng(42),evaluation=True),'evaluation sentinel')
        self.assertEqual(calls,[{'evaluation':True}])
        leaves,stats,events=adapter(self.tasks,engine,self.tokenizer,self.cfg,'patr',np.random.default_rng(42))
        self.assertEqual(stats['episodes'],12);self.assertEqual(stats['score_prunes'],0)
        self.assertEqual(sum(b.ntokens for b in leaves)>=stats['generated_tokens'],True)
        self.assertEqual(len(core.training_weights(leaves,2)),12)
        for ti in range(2):self.assertEqual(sum(b.task_id==ti for b in leaves),6)
    def test_remaining_budget_and_skipped_replay(self):
        engine=NS(llm_engine=Backend(self.tokenizer));b=core.start_branch(self.tasks[0],0,self.tokenizer,self.cfg,'0');b.order=(0,)
        from cert_collect import stream
        cfg=dict(self.cfg,max_context_tokens=len(b.ids))
        done,_,cost,_=stream(core,[b],self.tasks,engine,self.tokenizer,cfg,123,capture=False)
        self.assertEqual(done[0].terminal,'context_limit');self.assertEqual(cost['generated_tokens'],0)

    def test_queue_pid_identity(self):
        import os
        from interleave_queue import identity
        self.assertIsInstance(identity(os.getpid()),int)
        self.assertIsNone(identity(99999999))

    def test_g2_gate_exact_comparison(self):
        from g2_gate import decision
        b={100:69.75,125:70.25,150:71.75};u={s:v+1 for s,v in b.items()};r={s:v-.5 for s,v in b.items()}
        self.assertTrue(all(decision(u,r,b,.08,.07,.0721875).values()))
        self.assertFalse(all(decision(u,r,b,.0721,.07,.0721875).values()))
        self.assertFalse(all(decision(r,r,b,.08,.07,.0721875).values()))

if __name__=='__main__':unittest.main()
