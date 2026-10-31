"""Read-only second-turn probe. Does not modify baseline scripts/configs/results."""
import argparse, collections, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from patr_core import FrozenLakeAgent, make_env, distance, atomic_json
BASE=ROOT/'runs/audit';OUT=BASE/'state_probe_hist1'
MANIFEST_SHA='cd0cdc2d39c164a5c79eef6a1a635a59f156dc73daf1c791146daef68b944e08'
DIRECTIONS={1:'Left',2:'Down',3:'Right',4:'Up'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest():
    p=BASE/'state_manifest.json';assert sha(p)==MANIFEST_SHA
    return json.loads(p.read_text())['states']
def protected_hashes():
    paths=[ROOT/'train.py',ROOT/'patr_core.py',ROOT/'scripts/state_probe.py',BASE/'state_manifest.json']
    paths+=list((ROOT/'configs').rglob('*'))+list((BASE/'state_probe').rglob('*'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if p.is_file()}
def build_history(s):
    import numpy as np
    env=make_env(s['task'],dict(is_slippery=False,max_turns=10));target=s['state'];r,c=divmod(target,env.ncol);candidates=[]
    for action,(dr,dc) in enumerate([(0,-1),(1,0),(0,1),(-1,0)],1):
        pr,pc=r-dr,c-dc
        if not (0<=pr<env.nrow and 0<=pc<env.ncol) or env.desc[pr,pc]==b'H':continue
        p=pr*env.ncol+pc;env.s=p
        obs,reward,done,info=env.step(action)
        # Goal tiles are absorbing: non-hole alone does not guarantee a true predecessor.
        if int(env.s)==target and not done and reward==0 and info.get('action_is_effective'):
            candidates.append((p,action))
    if not candidates:return None
    seed=int.from_bytes(hashlib.sha256(str(s['id']).encode()).digest()[:8],'big')
    p,action=candidates[int(np.random.default_rng(seed).integers(len(candidates)))]
    env=make_env(s['task'],dict(is_slippery=False,max_turns=10));env.s=p
    agent=FrozenLakeAgent(max_steps=10);agent.update_from_env(env.render(),0.,False,{})
    text='```'+DIRECTIONS[action]+'```';parsed=int(agent.update_from_model(text).action);assert parsed==action
    obs,reward,done,info=env.step(parsed);assert int(env.s)==target and not done and reward==0
    agent.update_from_env(obs,reward,done,info)
    assert agent.step==1 and len(agent.chat_completions)==4
    assert 'Current Observation (1)' in agent.chat_completions[-1]['content']
    assert 'steps remaining is 9' in agent.chat_completions[-1]['content']
    assert distance(env)==s['shortest_distance']
    assert agent.chat_completions[0]==s['messages'][0]
    return dict(s,messages=agent.chat_completions,predecessor=p,history_action=action,history_response=text,predecessor_seed=seed,predecessor_candidates=[list(x) for x in candidates])
def prepare():
    OUT.mkdir(parents=True,exist_ok=True);states=manifest();histories=[];skipped=[]
    for s in states:
        h=build_history(s)
        if h is None:skipped.append(dict(id=s['id'],reason='No adjacent nonterminal safe predecessor can actually move into target'))
        else:histories.append(h)
    value=dict(manifest_sha256=MANIFEST_SHA,n_manifest=len(states),eligible_n=len(histories),skipped=skipped,states=histories,protocol='SHA256(str(state id)) first8bytes big-endian seed; numpy default_rng chooses from action-order1..4 real predecessors. Same FrozenLakeAgent updates and real env.step; supplied correct one-action history; secondturn has9 remainingsteps. No future actions/rewards in prompt.')
    dest=OUT/'history_manifest.json'
    if dest.exists():assert json.loads(dest.read_text())==value
    else:atomic_json(dest,value)
    guard=OUT/'protected_inputs_sha256.json';current=protected_hashes()
    if guard.exists():assert json.loads(guard.read_text())==current
    else:atomic_json(guard,current)
    atomic_json(OUT/'cpu_validation.json',dict(status='passed',n_states=len(states),eligible_n=len(histories),skipped_n=len(skipped),real_transition_checks=len(histories),history_manifest_sha256=sha(dest),protected_file_count=len(current),cuda_initialized=False))
    print('CPU history validation',len(histories),'eligible,',len(skipped),'skipped',flush=True)
def stats(rows):
    n=len(rows);parsed=[r for r in rows if r['parsed_legal']];accepted=[r for r in rows if r['completed_legal']]
    pct=lambda v,d:100*v/d if d else None
    return dict(n=n,legal_n=len(parsed),format_legal_pct=pct(len(parsed),n),optimal_given_legal_pct=pct(sum(r['parsed_optimal'] for r in parsed),len(parsed)),optimal_all_states_pct=pct(sum(r['parsed_optimal'] for r in rows),n),completed_legal_n=len(accepted),completed_legal_pct=pct(len(accepted),n),optimal_given_completed_legal_pct=pct(sum(r['parsed_optimal'] for r in accepted),len(accepted)),completed_optimal_all_states_pct=pct(sum(r['parsed_optimal'] for r in accepted),n),finish_reasons=dict(collections.Counter(r['finish_reason'] for r in rows)))
def normalize(row,state):
    _,act=FrozenLakeAgent()._parse_model_response(row['response']);act=int(act);legal=act in (1,2,3,4)
    return dict(row,parsed_action=act,parsed_legal=legal,parsed_optimal=act in state['optimal_actions'],completed_legal=legal and row['finish_reason']!='length',train_geometry_seen=state['train_geometry_seen'])
def make_engine(model):
    os.environ.update(VLLM_USE_V1='0',VLLM_ATTENTION_BACKEND='XFORMERS',VLLM_USE_FLASHINFER_SAMPLER='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',WANDB_MODE='offline')
    inc=ROOT/'artifacts/build-deps/usr/include'
    if (inc/'python3.12/Python.h').exists():os.environ['C_INCLUDE_PATH']=os.pathsep.join([str(inc/'python3.12'),str(inc),os.environ.get('C_INCLUDE_PATH','')])
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM
    torch.set_num_threads(8);tok=AutoTokenizer.from_pretrained(model,local_files_only=True);tok.pad_token=tok.eos_token
    cfg=json.loads((ROOT/'configs/tree_random400.json').read_text())
    engine=LLM(model=model,tokenizer=model,dtype='bfloat16',tensor_parallel_size=1,distributed_executor_backend='uni',gpu_memory_utilization=cfg['vllm_memory_fraction'],max_model_len=cfg['max_context_tokens'],max_num_seqs=cfg['vllm_max_sequences'],max_num_batched_tokens=4096,enable_chunked_prefill=True,enable_prefix_caching=True,disable_async_output_proc=True,seed=42,enforce_eager=True,disable_log_stats=True)
    return engine,tok

def generate(engine,tok,states,dest,model,label,mode,cap,skipped,load_seconds):
    from vllm import SamplingParams
    dest.mkdir(parents=True,exist_ok=False);start=time.monotonic();records=[]
    identity=dict(model=model,label=label,mode=mode,manifest_sha256=MANIFEST_SHA,history_manifest_sha256=sha(OUT/'history_manifest.json'),temperature=0,top_p=1,max_tokens=cap,seed=42,n_manifest=1000,n_evaluated=len(states),skipped=skipped,remaining_steps=9 if mode=='hist1' else 10,read_only=True,training_or_selection=False,model_load_seconds=load_seconds,parameter_updates=0,source_sha256=sha(Path(__file__)))
    atomic_json(dest/'identity.json',identity)
    params=SamplingParams(temperature=0.,top_p=1.,max_tokens=cap,seed=42,stop_token_ids=[tok.eos_token_id])
    with (dest/'responses.jsonl').open('w') as f:
        for off in range(0,len(states),64):
            chunk=states[off:off+64];prompts=[{'prompt_token_ids':tok.apply_chat_template(s['messages'],tokenize=True,add_generation_prompt=True)} for s in chunk]
            for s,result in zip(chunk,engine.generate(prompts,params,use_tqdm=False)):
                o=result.outputs[0];row=normalize(dict(id=s['id'],uid=s['task']['uid'],state=s['state'],response=o.text,tokens=len(o.token_ids),finish_reason=o.finish_reason),s)
                records.append(row);f.write(json.dumps(row)+'\n')
            f.flush();atomic_json(dest/'status.json',dict(status='running',completed=len(records),total=len(states)));print(label,mode,len(records),flush=True)
    summary=dict(all_states=stats(records),seen_training_geometry=stats([r for r in records if r['train_geometry_seen']]),unseen_training_geometry=stats([r for r in records if not r['train_geometry_seen']]),skipped_n=len(skipped),n_manifest=1000,seconds=time.monotonic()-start,total_seconds_including_load=time.monotonic()-start+load_seconds,model_load_seconds=load_seconds,max_tokens=cap,generated_tokens=sum(r['tokens'] for r in records),note='Fixed-state one-action diagnostic, not game success. Syntactic parsing matches old probe; completed-action metrics additionally reject length-limit outputs. Hist1 also changes remaining budget10->9 and cap10240->1024, so it is not a single-factor causal contrast. Never used for training or checkpoint selection.')
    atomic_json(dest/'summary.json',summary);atomic_json(dest/'status.json',dict(status='complete',completed=len(records),total=len(states)))
def table():
    states=manifest();byid={s['id']:s for s in states};h=json.loads((OUT/'history_manifest.json').read_text());eligible={s['id'] for s in h['states']};output=[]
    for label in ['old_patr','sft','sft_grpo400','sft_patr400','sft_patr500','sft_tree_random400','base']:
        hist=OUT/label/'summary.json'
        if not hist.exists():continue
        old=BASE/'state_probe'/label
        if label=='sft_tree_random400':old=OUT/label/'no_history_reference'
        row=dict(label=label,history=json.loads(hist.read_text()),no_history=None,no_history_on_hist_eligible=None)
        if (old/'responses.jsonl').exists() and (old/'summary.json').exists():
            rs=[normalize(json.loads(s),byid[json.loads(s)['id']]) for s in (old/'responses.jsonl').read_text().splitlines()]
            row['no_history']=dict(all_states=stats(rs),seen_training_geometry=stats([r for r in rs if r['train_geometry_seen']]),unseen_training_geometry=stats([r for r in rs if not r['train_geometry_seen']]),seconds=json.loads((old/'summary.json').read_text())['seconds'],max_tokens=10240)
            row['no_history_on_hist_eligible']=stats([r for r in rs if r['id'] in eligible])
        output.append(row)
    result=dict(models=output,skipped=h['skipped'],skipped_n=len(h['skipped']),n_manifest=1000,eligible_n=len(eligible),manifest_sha256=MANIFEST_SHA,limits='Single-step diagnostics, not trajectory success; supplied correct history; budget9vs10 and cap1024vs10240 differ. Full and eligible-matched no-history metrics reported; no training/selection.',gpu_overlap_evidence='scheduler_events.jsonl and scheduler_status.json; no GPU launched until original queue complete and its exclusive lock acquired')
    atomic_json(OUT/'summary_table.json',result)
    def val(x):return 'N/A' if x is None else f'{x:.2f}'
    lines=['| Model | No history legal % | Hist1 legal % | No history optimal given legal % | Hist1 optimal given legal % | No history optimal all % | Hist1 optimal all % | Hist1 seconds incl load |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in output:
        a=r['no_history_on_hist_eligible'] or {};b=r['history']['all_states']
        cells=[r['label']]+[val(x) for x in [a.get('format_legal_pct'),b['format_legal_pct'],a.get('optimal_given_legal_pct'),b['optimal_given_legal_pct'],a.get('optimal_all_states_pct'),b['optimal_all_states_pct'],r['history']['total_seconds_including_load']]];lines.append('| '+' | '.join(cells)+' |')
    (OUT/'summary_table.md').write_text('\n'.join(lines)+'\n\nSyntactic metrics on identical history-eligible states shown; full1000 no-history, completed-action and seen/unseen geometry breakdowns are in JSON. Caps: no-history10240, hist1 1024. Remaining steps:10vs9. Not game success. Skipped='+str(len(h['skipped']))+'.\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--model');p.add_argument('--label');p.add_argument('--fresh-reference',action='store_true');a=p.parse_args()
    if a.prepare:prepare();return
    if a.summarize:table();return
    assert a.model and a.label
    # Only the waiting supervisor may initialize CUDA, after it owns the original queue lock.
    assert os.environ.get('HIST1_GPU_LEASE')=='queue_complete_lock_held'
    import fcntl
    fd=int(os.environ['HIST1_GPU_LEASE_FD']);assert os.fstat(fd).st_ino==(ROOT/'runs/control_queue/supervisor.lock').stat().st_ino
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert protected_hashes()==json.loads((OUT/'protected_inputs_sha256.json').read_text())
    hist=json.loads((OUT/'history_manifest.json').read_text());t=time.monotonic();engine,tok=make_engine(a.model);loaded=time.monotonic()-t
    if not (OUT/a.label/'summary.json').exists():generate(engine,tok,hist['states'],OUT/a.label,a.model,a.label,'hist1',1024,hist['skipped'],loaded)
    if a.fresh_reference and not (OUT/a.label/'no_history_reference/summary.json').exists():generate(engine,tok,manifest(),OUT/a.label/'no_history_reference',a.model,a.label,'no_history_reference',10240,[],0.)
    assert protected_hashes()==json.loads((OUT/'protected_inputs_sha256.json').read_text())
    table()
if __name__=='__main__':main()
