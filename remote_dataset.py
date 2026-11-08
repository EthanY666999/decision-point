"""Standalone FLH teacher-data store. CPU only; never imports training or model code."""
import collections,copy,datetime,fcntl,gzip,hashlib,json,os,random,shutil,sys
from pathlib import Path
ROOT=Path('/home/vipuser/golden-flh-trace-20261108')
SOURCE=Path('/home/vipuser/decision-point-q35-hard')
MODEL='gpt-6.1-sol';TARGET=1000;MAX_ATTEMPTS=1600
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,d):
 p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
STATIC={}
def read(p):
 if p in ('maps.json','data.json','candidates.json'):
  if p not in STATIC:STATIC[p]=json.loads((ROOT/p).read_text())
  return STATIC[p]
 return json.loads((ROOT/p).read_text())
def init():
 if (ROOT/'manifest.json').exists():return read('status.json')
 ROOT.mkdir(parents=True,exist_ok=True)
 shutil.copytree(SOURCE/'code/vendor/rllm_source',ROOT/'source/rllm_source',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
 droot=SOURCE/'results/q35_hard_datasets/flh';mf=json.loads((droot/'manifest.json').read_text())
 for n in ['data.json','maps.json']:
  assert sha((droot/n).read_bytes())==mf['files'][n];shutil.copyfile(droot/n,ROOT/n)
 data=read('data.json');maps=read('maps.json');heldout=data['validation']+data['test']+data['supplemental']
 blocked={t['map_sha256'] for t in heldout};uids={t['uid'] for t in heldout}
 assert len(data['train'])==10000 and len(heldout)==700
 candidates=[];seen=set()
 for t in data['train']:
  h=sha('\n'.join(maps[t['uid']]).encode());assert h==t['map_sha256']
  assert t['split']=='train' and 9<=t['size']<=12 and 7<=t['shortest_path_length']<=16
  assert h not in blocked and t['uid'] not in uids and h not in seen
  seen.add(h);candidates.append(t)
 random.Random(20261108).shuffle(candidates)
 save('candidates.json',candidates)
 save('manifest.json',dict(teacher_model=MODEL,teacher_fallback=None,reasoning_effort='medium',target=TARGET,max_attempts=MAX_ATTEMPTS,pilot_size=32,pilot_min_success_fraction=.8,
  protocol='Teacher-only complete action plans from fully observed deterministic training maps; each action executed by actual frozen environment; student messages reconstructed from actual observations. Not per-turn teacher queries.',
  student_response_style='one fenced action per environment turn; original agent prompt retained',map_source=str(droot),dataset_files_sha256=mf['files'],heldout_maps_excluded=700,
  candidate_order_seed=20261108,max_turns=20,selection='seeded shuffle of train only; retain successful complete plans; failures preserved; shortest-path optimality not required',
  no_bfs_actions=True,bfs_use='existing shortest-path metadata used for audit only; never shown to teacher or substituted into teacher plans',
  source_sha256={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in (ROOT/'source').rglob('*') if p.is_file()},
  model_provenance='explicit CLI --model; CLI events may not separately expose provider model identity',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 save('status.json',dict(state='prepared',accepted=0,attempted=0,target=TARGET))
 return read('status.json')
def episode_paths():return sorted((ROOT/'episodes').glob('*.json'))
def stats():
 rs=[json.loads(p.read_text()) for p in episode_paths()];return rs,sum(r['accepted'] for r in rs)
def request():
 rs,n=stats();used={r['task']['uid'] for r in rs}
 if n>=TARGET:return dict(complete=True,accepted=n)
 if len(rs)>=MAX_ATTEMPTS:raise RuntimeError('attempt budget exhausted')
 if (ROOT/'pending_request.json').exists():return read('pending_request.json')
 cand=[t for t in read('candidates.json') if t['uid'] not in used];size=min(32,TARGET-n,MAX_ATTEMPTS-len(rs));cand=cand[:size];maps=read('maps.json')
 bid=f'b{len(rs):04d}';req=dict(batch_id=bid,instruction='Solve each independent deterministic FrozenLake map. S=start G=goal F=safe H=hole. One orthogonal move per action; Up decreases row, Down increases row, Left decreases column, Right increases column. Produce a complete safe plan reaching G within 20 moves; prefer short safe routes. Each game is independent. No tools, code, search, delegation, or invented observations. Return one id/actions entry per game, JSON only.',games=[dict(id=t['uid'],rows=maps[t['uid']]) for t in cand])
 save('pending_request.json',req);save('status.json',dict(state='awaiting_teacher',batch=bid,accepted=n,attempted=len(rs),target=TARGET,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 return req
def env_agent(t):
 sys.path.insert(0,str(ROOT/'source/rllm_source'))
 from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
 from rllm.agents.frozenlake_agent import FrozenLakeAgent
 e=FrozenLakeEnv(desc=read('maps.json')[t['uid']],seed=t['seed'],is_slippery=False,max_steps=20);e.reset()
 a=FrozenLakeAgent(max_steps=20);a.update_from_env(e.render(),0.,False,{})
 return e,a
def execute(t,actions,batch):
 e,a=env_agent(t);steps=[];terminal='plan_exhausted';done=False
 if not isinstance(actions,list) or not 1<=len(actions)<=20 or not all(x in ['Up','Down','Left','Right'] for x in actions):
  return dict(task=t,map_hash=t['map_sha256'],map=read('maps.json')[t['uid']],teacher_batch=batch,requested_teacher_model=MODEL,planned_actions=actions,steps=[],messages=a.chat_completions,terminal='bad_plan_format',accepted=False,shortest_distance=t['shortest_path_length'],no_op_count=0)
 for i,word in enumerate(actions):
  if done:terminal='actions_after_terminal';break
  response='```'+word+'```';messages=copy.deepcopy(a.chat_completions);before=int(e.s);act=int(a.update_from_model(response).action)
  obs,reward,done,info=e.step(act)
  steps.append(dict(turn=i+1,messages_before=messages,response=response,action=act,state_before=before,state_after=int(e.s),observation_after=obs,reward=float(reward)))
  if done:terminal='success' if e.success() else 'hole'
  else:a.update_from_env(obs,reward,done,info)
 accepted=done and e.success() and terminal=='success'
 return dict(task=t,map_hash=t['map_sha256'],map=read('maps.json')[t['uid']],teacher_batch=batch,requested_teacher_model=MODEL,planned_actions=actions,steps=steps,messages=a.chat_completions,terminal=terminal,accepted=accepted,shortest_distance=t['shortest_path_length'],no_op_count=sum(s['state_before']==s['state_after'] for s in steps))
def ingest(packet):
 if not (ROOT/'pending_request.json').exists():
  bid=packet['request']['batch_id'];p=ROOT/'provenance'/f"{bid}-{packet.get('call_id','first')}.json"
  assert p.exists() and json.loads(p.read_text())==packet,'unrecognized delivery retry'
  for game in packet['request']['games']:
   ep=ROOT/'episodes'/(game['id']+'.json');assert ep.exists() and json.loads(ep.read_text())['teacher_batch']==bid,'incomplete prior delivery'
  return read('status.json')
 req=read('pending_request.json');bid=req['batch_id'];assert packet['request']==req and packet['requested_model']==MODEL
 save('provenance/'+bid+'-'+packet.get('call_id','first')+'.json',packet)
 if packet['returncode']!=0:raise RuntimeError('teacher exited unsuccessfully; raw call retained')
 events=[json.loads(x) for x in packet['stdout'].splitlines() if x.strip()]
 assert any(x.get('type')=='turn.completed' for x in events),'teacher not complete'
 for e in events:
  item=e.get('item',{});assert item.get('type') not in ['command_execution','mcp_tool_call','web_search','collab_tool_call'],'teacher used tools'
 texts=[e['item']['text'] for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='agent_message']
 answers=json.loads(texts[-1])['answers'];ids=[x['id'] for x in answers];assert len(ids)==len(set(ids)) and set(ids)=={g['id'] for g in req['games']}
 tasks={t['uid']:t for t in read('candidates.json')}
 for answer in answers:
  t=tasks[answer['id']];out=execute(t,answer['actions'],bid)
  if out['accepted']:
   again=execute(t,answer['actions'],bid);assert again==out,'independent replay mismatch'
  save('episodes/'+t['uid']+'.json',out)
 (ROOT/'pending_request.json').unlink();rs,n=stats()
 if len(rs)==32:
  pilot=dict(attempted=32,accepted=n,passed=n>=26,threshold=.8);save('pilot.json',pilot)
  if not pilot['passed']:save('status.json',dict(state='pilot_quality_failed',**pilot,target=TARGET));return read('status.json')
 save('status.json',dict(state='batch_complete',accepted=n,attempted=len(rs),target=TARGET,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 export();return read('status.json')
def export():
 rs,n=stats();accepted=[r for r in rs if r['accepted']];out=ROOT/'release';out.mkdir(exist_ok=True)
 with gzip.open(out/'golden_traces.jsonl.gz','wt',encoding='utf-8') as f,gzip.open(out/'sft_messages.jsonl.gz','wt',encoding='utf-8') as s:
  for r in accepted:
   f.write(json.dumps(r,ensure_ascii=False)+'\n');s.write(json.dumps(dict(task_uid=r['task']['uid'],map_hash=r['map_hash'],messages=r['messages']),ensure_ascii=False)+'\n')
 return n
def verify():
 rs,n=stats();accepted=[r for r in rs if r['accepted']];assert len({r['map_hash'] for r in accepted})==n
 for r in accepted:assert execute(r['task'],r['planned_actions'],r['teacher_batch'])==r
 assert n==TARGET
 export();summary=dict(complete=True,independently_replayed=n,target=TARGET,attempted=len(rs),success_rate=n/len(rs),mean_steps=sum(len(r['steps']) for r in accepted)/n,optimal_count=sum(len(r['steps'])==r['shortest_distance'] for r in accepted),teacher_model=MODEL,no_fallback=True,
  size_counts=dict(collections.Counter(r['task']['size'] for r in accepted)),shortest_path_counts=dict(collections.Counter(r['shortest_distance'] for r in accepted)),detour_count=sum(r['task']['requires_detour'] for r in accepted),files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in (ROOT/'release').glob('*.gz')})
 save('release/summary.json',summary);save('status.json',dict(state='complete',accepted=n,attempted=len(rs),target=TARGET,independent_final_replay=True,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()));return summary
if __name__=='__main__':
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'dataset.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  cmd=sys.argv[1]
  result=init() if cmd=='prepare' else request() if cmd=='request' else ingest(json.load(sys.stdin)) if cmd=='ingest' else verify() if cmd=='verify' else read('status.json')
  print(json.dumps(result,ensure_ascii=False))
