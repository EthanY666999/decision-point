"""Frozen shared-state greedy inference; read-only model diagnostics."""
import argparse,hashlib,json,sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train import build_models
from patr_core import atomic_json,FrozenLakeAgent
from vllm import SamplingParams
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--label',required=True);a=p.parse_args()
manifest=ROOT/'runs/audit/state_manifest.json';identity=json.loads((manifest.with_name('state_manifest_identity.json')).read_text());assert hashlib.sha256(manifest.read_bytes()).hexdigest()==identity['sha256']
states=json.loads(manifest.read_text())['states'];cfg=json.loads((ROOT/'configs/tree_random400.json').read_text());cfg['model']=a.model
out=ROOT/'runs/audit/state_probe'/a.label;out.mkdir(parents=True,exist_ok=False)
atomic_json(out/'identity.json',dict(model=a.model,manifest_sha256=identity['sha256'],decoding='greedy temperature0; top_p1; max_response_tokens unchanged10240',n=len(states)))
t0=time.monotonic();actor,ref,engine,tok=build_models(cfg,42)
params=SamplingParams(temperature=0.,top_p=1.,max_tokens=cfg['max_response_tokens'],seed=42)
records=[];parser=FrozenLakeAgent(max_steps=10)
with (out/'responses.jsonl').open('w') as f:
    for offset in range(0,len(states),64):
        chunk=states[offset:offset+64];prompts=[{'prompt_token_ids':tok.apply_chat_template(s['messages'],tokenize=True,add_generation_prompt=True)} for s in chunk]
        outputs=engine.generate(prompts,params,use_tqdm=False)
        for s,res in zip(chunk,outputs):
            generated=res.outputs[0];_,act=parser._parse_model_response(generated.text);act=int(act);legal=act in [1,2,3,4]
            r=dict(id=s['id'],uid=s['task']['uid'],state=s['state'],action=act,legal=legal,optimal=act in s['optimal_actions'],optimal_actions=s['optimal_actions'],response=generated.text,finish_reason=generated.finish_reason,tokens=len(generated.token_ids),train_geometry_seen=s['train_geometry_seen'])
            records.append(r);f.write(json.dumps(r)+'\n')
        f.flush();atomic_json(out/'status.json',dict(status='running',completed=len(records),total=len(states)));print(a.label,len(records),flush=True)
def stats(rs):
    n=len(rs);legal=[r for r in rs if r['legal']]
    return dict(n=n,legal_n=len(legal),format_legal_pct=100*len(legal)/n if n else None,optimal_given_legal_pct=100*sum(r['optimal'] for r in legal)/len(legal) if legal else None,optimal_all_states_pct=100*sum(r['optimal'] for r in rs)/n if n else None)
summary=dict(all_states=stats(records),unseen_training_geometry=stats([r for r in records if not r['train_geometry_seen']]),seconds=time.monotonic()-t0,generated_tokens=sum(r['tokens'] for r in records),finish_reasons=dict(collections.Counter(r['finish_reason'] for r in records)),note='Fresh state-only contexts, not natural on-policy histories. Conditional optimal accuracy requires legal denominator; comparing models with different legality coverage is not a causal planning estimate. Test diagnostic not used to select training.')
atomic_json(out/'summary.json',summary);atomic_json(out/'status.json',dict(status='complete',completed=len(records),total=len(states)));print(json.dumps(summary),flush=True)
