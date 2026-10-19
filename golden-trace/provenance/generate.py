import collections, copy, datetime, fcntl, hashlib, json, os, pathlib, random, re, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT.parent.parent
sys.path.insert(0, str(ROOT/'vendor/rllm_source'))
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.agents.frozenlake_agent import FrozenLakeAgent

MODEL = 'gpt-6.1-sol'
DATA = WORK/'output/patr-v1-tree-closeout-20261017/patr-v1-tree-closeout-20261017/source/artifacts/data.json'
EXTRA = WORK/'output/patr-eval500-20261017/patr-supplemental500-20261017/tasks.json'
ACTION = re.compile(r'^```(Up|Down|Left|Right)```$')

def sha(b): return hashlib.sha256(b).hexdigest()
def save(name, value):
    p=ROOT/name; p.parent.mkdir(parents=True, exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); q.replace(p)
def append(name, value):
    with (ROOT/name).open('a') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
def event(kind, **kw):
    x=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=kind,**kw)
    append('events.jsonl',x); print(json.dumps(x),flush=True)
def env(task, max_steps=10):
    e=FrozenLakeEnv(size=task['size'],p=task['p'],seed=task['seed'],is_slippery=False,max_steps=max_steps)
    e.reset(); return e
def maprows(e): return [''.join(x.decode() for x in r) for r in e.desc]
def maphash(e): return sha('\n'.join(maprows(e)).encode())
def distance(e):
    todo=collections.deque([(int(e.s),0)]); seen={int(e.s)}
    while todo:
        p,d=todo.popleft();r,c=divmod(p,e.ncol)
        if e.desc[r,c]==b'G': return d
        for dr,dc in [(0,-1),(1,0),(0,1),(-1,0)]:
            rr,cc=r+dr,c+dc; q=rr*e.ncol+cc
            if 0<=rr<e.nrow and 0<=cc<e.ncol and q not in seen and e.desc[rr,cc]!=b'H':
                seen.add(q);todo.append((q,d+1))
    return None

def prepare():
    data=json.loads(DATA.read_text()); extra=json.loads(EXTRA.read_text())
    blocked_tasks=data['validation']+data['test']+extra
    blocked_uid={t['uid'] for t in blocked_tasks}
    blocked={maphash(env(t)) for t in blocked_tasks}
    # Also exclude the separately evaluated map-generation max_steps=8 variant.
    blocked.update(maphash(env(t,8)) for t in data['test'])
    candidates=[]; seen=set(); counts=collections.Counter()
    for t in data['train']:
        e=env(t); h=maphash(e); d=distance(e)
        if t['uid'] in blocked_uid or h in blocked: counts['evaluation_overlap']+=1;continue
        if h in seen: counts['duplicate_train_map']+=1;continue
        if d is None or d>10: counts['not_solvable_within_10_turns']+=1;continue
        seen.add(h);candidates.append(dict(task=t,map_hash=h,shortest_distance=d,map=maprows(e)))
    buckets=collections.defaultdict(list)
    for x in candidates:buckets[(x['task']['size'],x['shortest_distance'])].append(x)
    rng=random.Random(42)
    for xs in buckets.values():rng.shuffle(xs)
    ordered=[]
    while any(buckets.values()):
        for k in sorted(buckets):
            if buckets[k]:ordered.append(buckets[k].pop())
    save('candidates.json',ordered)
    manifest=dict(dataset_sha256=sha(DATA.read_bytes()),supplement500_sha256=sha(EXTRA.read_bytes()),
       train_tasks=len(data['train']),eligible_unique_maps=len(ordered),exclusions=dict(counts),
       blocked_geometry_count=len(blocked),teacher_model=MODEL,teacher_fallback=None,
       generation_protocol='batched independent conversations; one action per actual environment turn',
       response_style='action-only fenced response; teacher-only wrapper requests this style; student legacy prompt unchanged',
       pilot_tasks=32,pilot_min_success_fraction=.8,target_valid=1000,max_tasks=2000,
       source_sha256={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in (ROOT/'vendor').rglob('*.py')},
       no_training=True,created=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save('manifest.json',manifest);event('prepared',**{k:manifest[k] for k in ['train_tasks','eligible_unique_maps','exclusions']})

def teacher(games, batch, turn):
    request_id=f'b{batch:04d}-t{turn:02d}'
    request=dict(instruction='Act as the teacher policy for independent FrozenLake games. For each id, follow its conversation and return only the NEXT action, exactly one of ```Up```, ```Down```, ```Left```, ```Right```. No prose, no tools, no code execution, no delegation. The local environment executes actions; never invent observations. Each game is independent. Return JSON {"answers":[{"id":"...","response":"```Right```"}]}.',
        games=[dict(id=g['id'],messages=g['agent'].chat_completions) for g in games])
    prompt=json.dumps(request,ensure_ascii=False); save('requests/'+request_id+'.json',request)
    out=ROOT/'requests'/f'{request_id}.response.json'
    schema=ROOT/'response_schema.json'
    cmd=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','--model',MODEL,
         '-c','model_reasoning_effort="medium"','--disable','shell_tool','--disable','multi_agent',
         '--sandbox','read-only','-C',str(ROOT),'--json','--output-schema',str(schema),'-o',str(out),'-']
    process_env=os.environ.copy()
    # Honor the user's existing macOS HTTP proxy, scoped to this subprocess only.
    proxy=subprocess.check_output(['scutil','--proxy'],text=True)
    host=re.search(r'HTTPSProxy\s*:\s*(\S+)',proxy);port=re.search(r'HTTPSPort\s*:\s*(\d+)',proxy)
    if host and port and re.search(r'HTTPSEnable\s*:\s*1',proxy):
        process_env['HTTPS_PROXY']=process_env['HTTP_PROXY']=f'http://{host.group(1)}:{port.group(1)}'
    save('requests/'+request_id+'.invocation.json',dict(argv=cmd,requested_model=MODEL,model_provenance='explicit CLI --model; provider model identifier is not separately exposed by CLI JSON events',prompt_sha256=sha(prompt.encode())))
    start=time.monotonic()
    with (ROOT/'requests'/f'{request_id}.events.jsonl').open('w') as stdout, (ROOT/'requests'/f'{request_id}.stderr.log').open('w') as stderr:
        result=subprocess.run(cmd,input=prompt,text=True,stdout=stdout,stderr=stderr,env=process_env,timeout=300)
    events=[json.loads(s) for s in (ROOT/'requests'/f'{request_id}.events.jsonl').read_text().splitlines() if s.strip()]
    assert result.returncode==0 and out.exists(), ('teacher_call_failed',request_id,result.returncode)
    completed=[e for e in events if e.get('type')=='turn.completed'];assert completed,('no_turn_completion',request_id)
    for e in events:
        item=e.get('item',{})
        assert item.get('type') not in ['command_execution','mcp_tool_call','web_search','collab_tool_call'],('unexpected_teacher_tool',e)
    parsed=json.loads(out.read_text());answers=parsed['answers']
    assert len(answers)==len(games) and {x['id'] for x in answers}=={g['id'] for g in games}, ('answer_ids',request_id)
    assert all(ACTION.fullmatch(x['response']) for x in answers),('action_format',request_id)
    meta=dict(request_id=request_id,requested_model=MODEL,prompt_sha256=sha(prompt.encode()),
              response_sha256=sha(out.read_bytes()),usage=completed[-1].get('usage'),seconds=time.monotonic()-start,
              cli_thread_id=next((x.get('thread_id') for x in events if x.get('type')=='thread.started'),None))
    append('requests.jsonl',meta);return {x['id']:x['response'] for x in answers},meta

def replay(record):
    e=env(record['task']);a=FrozenLakeAgent(max_steps=10);a.update_from_env(e.render(),0.,False,{})
    assert maphash(e)==record['map_hash']
    assert 0<len(record['steps'])<=10
    done=False
    for step in record['steps']:
        assert not done and step['messages_before']==a.chat_completions
        assert ACTION.fullmatch(step['response'])
        action=int(a.update_from_model(step['response']).action)
        observation,reward,done,info=e.step(action)
        assert action==step['action'] and int(e.s)==step['state_after'] and observation==step['observation_after'] and float(reward)==step['reward']
        if not done:a.update_from_env(observation,reward,done,info)
    assert e.success() and done and a.chat_completions==record['messages']
    return True

def run(target=1000):
    lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert target<=1000
    candidates=json.loads((ROOT/'candidates.json').read_text())
    schema=dict(type='object',properties={'answers':dict(type='array',items=dict(type='object',properties={'id':{'type':'string'},'response':{'type':'string'}},required=['id','response'],additionalProperties=False))},required=['answers'],additionalProperties=False)
    save('response_schema.json',schema)
    done_records=[json.loads(p.read_text()) for p in sorted((ROOT/'episodes').glob('*.json'))] if (ROOT/'episodes').exists() else []
    used={x['task']['uid'] for x in done_records};valid=sum(x['accepted'] for x in done_records);attempted=len(done_records)
    remaining=[x for x in candidates if x['task']['uid'] not in used]
    batch=1+max([int(p.name.split('-')[0][1:]) for p in (ROOT/'requests').glob('b*.invocation.json')],default=0)
    try:
        while valid<target and remaining and attempted<2000:
            size=min(32,target-valid,2000-attempted)
            chosen=remaining[:size];remaining=remaining[size:];games=[]
            for c in chosen:
                e=env(c['task']);a=FrozenLakeAgent(max_steps=10);a.update_from_env(e.render(),0.,False,{})
                games.append(dict(**c,id=c['task']['uid'],env=e,agent=a,steps=[],terminal=None))
            save('status.json',dict(state='running',pid=os.getpid(),valid=valid,attempted=attempted,target=target,batch=batch,model=MODEL))
            event('batch_started',batch=batch,games=len(games),valid=valid)
            for turn in range(1,11):
                active=[g for g in games if g['terminal'] is None]
                if not active:break
                answers,meta=teacher(active,batch,turn)
                for g in active:
                    response=answers[g['id']];messages=copy.deepcopy(g['agent'].chat_completions)
                    action=int(g['agent'].update_from_model(response).action);before=int(g['env'].s)
                    observation,reward,done,info=g['env'].step(action)
                    g['steps'].append(dict(turn=turn,messages_before=messages,response=response,action=action,
                        state_before=before,state_after=int(g['env'].s),observation_after=observation,reward=float(reward),request_id=meta['request_id']))
                    if done:g['terminal']='success' if g['env'].success() else 'hole'
                    elif turn==10:g['terminal']='turn_limit'
                    if not done:g['agent'].update_from_env(observation,reward,done,info)
                    save('in_progress/'+sha(g['id'].encode())+'.json',dict(task=g['task'],steps=g['steps'],terminal=g['terminal']))
                event('turn_completed',batch=batch,turn=turn,active=len(active),seconds=round(meta['seconds'],2))
            for g in games:
                r={k:v for k,v in g.items() if k not in ['env','agent','id']}
                r.update(messages=g['agent'].chat_completions,requested_teacher_model=MODEL,accepted=False,
                         no_op_count=sum(s['state_before']==s['state_after'] for s in g['steps']))
                if r['terminal']=='success':r['accepted']=replay(r)
                save('episodes/'+sha(g['id'].encode())+'.json',r);done_records.append(r)
                valid+=int(r['accepted']);attempted+=1
            batch+=1
            save('status.json',dict(state='running',pid=os.getpid(),valid=valid,attempted=attempted,target=target,batch=batch,model=MODEL))
            if attempted>=32 and not (ROOT/'pilot.json').exists():
                first=done_records[:32];passed=sum(x['accepted'] for x in first)/32>=.8
                save('pilot.json',dict(tasks=32,accepted=sum(x['accepted'] for x in first),passed=passed,threshold=.8))
                assert passed,'pilot_quality_below_predeclared_80_percent'
        accepted=[r for r in done_records if r['accepted']]
        for r in accepted:replay(r)
        assert len({r['map_hash'] for r in accepted})==len(accepted)
        with (ROOT/'golden_traces.jsonl').open('w') as f, (ROOT/'sft_messages.jsonl').open('w') as s:
            for r in accepted:
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
                s.write(json.dumps(dict(task_uid=r['task']['uid'],map_hash=r['map_hash'],messages=r['messages']),ensure_ascii=False)+'\n')
        save('status.json',dict(state='complete' if valid>=target else 'incomplete',valid=valid,attempted=attempted,target=target,model=MODEL,
            dataset_sha256=sha((ROOT/'golden_traces.jsonl').read_bytes()),all_accepted_replayed=True))
        event('finished',valid=valid,attempted=attempted)
    except BaseException as ex:
        save('status.json',dict(state='paused_error',valid=valid,attempted=attempted,target=target,model=MODEL,error=repr(ex)))
        event('paused_error',error=repr(ex));raise

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run(int(sys.argv[2]) if len(sys.argv)>2 else 1000)
