"""Read-only dataset verification. No teacher API/CLI requests or training."""
from pathlib import Path
import hashlib,json,re,sys
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent

def digest(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()
for name,info in json.loads((ROOT/'CHECKSUMS.json').read_text())['files'].items():
 p=ROOT/name
 assert p.is_file() and digest(p)==info['sha256'] and p.stat().st_size==info['bytes'],name
manifest=json.loads((ROOT/'manifest.json').read_text())
for name,h in manifest['source_sha256'].items():
 assert digest(REPO/'code'/name)==h, f'Environment identity mismatch: {name}'
sys.path.insert(0,str(REPO/'code/vendor/rllm_source'))
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.agents.frozenlake_agent import FrozenLakeAgent
rows=[json.loads(l) for l in (ROOT/'golden_traces.jsonl').read_text().splitlines()]
sft=[json.loads(l) for l in (ROOT/'sft_messages.jsonl').read_text().splitlines()]
assert len(rows)==len(sft)==1000
assert len({x['map_hash'] for x in rows})==1000
by_uid={x['task_uid']:x for x in sft};assert len(by_uid)==len(sft)
for record in rows:
 t=record['task'];e=FrozenLakeEnv(size=t['size'],p=t['p'],seed=t['seed'],is_slippery=False,max_steps=10);e.reset()
 map_text='\n'.join(''.join(c.decode() for c in row) for row in e.desc)
 assert hashlib.sha256(map_text.encode()).hexdigest()==record['map_hash']
 a=FrozenLakeAgent(max_steps=10);a.update_from_env(e.render(),0.,False,{})
 done=False
 assert 0<len(record['steps'])<=10
 for s in record['steps']:
  assert not done and s['messages_before']==a.chat_completions
  assert re.fullmatch(r'```(Up|Down|Left|Right)```',s['response'])
  action=int(a.update_from_model(s['response']).action)
  obs,reward,done,_=e.step(action)
  assert action==s['action'] and int(e.s)==s['state_after'] and obs==s['observation_after'] and float(reward)==s['reward']
  if not done:a.update_from_env(obs,reward,done,{})
 assert done and e.success() and a.chat_completions==record['messages']
 target=by_uid[t['uid']]
 assert target['map_hash']==record['map_hash'] and target['messages']==record['messages']
assert digest(ROOT/'golden_traces.jsonl')==json.loads((ROOT/'summary.json').read_text())['generated_dataset_sha256']
print('PASS: hashes, frozen environment, 1000 successful replays, 1000 unique maps, and 1000 aligned SFT records.')
