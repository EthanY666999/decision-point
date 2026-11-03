"""Publish verified remote shards to main without storing a local dataset clone."""
import base64,fcntl,hashlib,json,pathlib,subprocess,time,sys
import generate as g
ROOT=g.ROOT; REPO='repos/EthanY666999/decision-point'; PREFIX='golden-trace-swe/2026-11-03-gpt-6.1-sol-r2egym'
SNAPSHOT=r'''
import pathlib,json,hashlib
r=pathlib.Path('/home/vipuser/golden-swe-trace-20261103')
rows=[json.loads(x) for x in (r/'golden_traces.jsonl').read_text().splitlines() if x.strip()]
assert len({x['instance_id'] for x in rows})==len(rows)
proof=[]
for e in rows:
 v=e['result'];assert v['accepted'] and v['matching_tests']==v['expected_tests']>0
 assert e['submitted'] and e['requested_teacher_model']=='gpt-6.1-sol'
 assert e['calibration']['reference_passed'] and not e['calibration']['baseline_passed']
 assert hashlib.sha256(v['patch'].encode()).hexdigest()==v['patch_sha256']
 d=r/'environment_states'/e['token'];report=json.loads((d/'independent_verification.json').read_text())
 assert report['passed'] and report['log_sha256']==hashlib.sha256((d/'independent_verification.log').read_bytes()).hexdigest()
 proof.append({'instance_id':e['instance_id'],'patch_sha256':v['patch_sha256'],'verification_log_sha256':report['log_sha256'],'expected_tests':v['expected_tests'],'matching_tests':v['matching_tests'],'expected_passing_tests':sum(x=='PASSED' for x in report['expected'].values()),'expected_nonpassing_tests':sum(x!='PASSED' for x in report['expected'].values())})
files={}
for i in range(0,len(rows),10):
 group=rows[i:i+10]
 files[f'trajectories/part-{i//10:04d}.jsonl']=''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in group)
 files[f'sft/part-{i//10:04d}.jsonl']=''.join(json.dumps({'instance_id':e['instance_id'],'messages':e['messages'],'teacher':'gpt-6.1-sol','source':'R2E-Gym training split','verification':e['result']},ensure_ascii=False)+'\n' for e in group)
files['verification.jsonl']=''.join(json.dumps(x)+'\n' for x in proof)
for name in ['generate.py','remote_runner.py','remote_env.py','test_acceptance.py','prepare_data.py','recovery_utils.py','remote_supervisor.py','test_recovery.py']:
 files['generator/'+name]=(r/name).read_text()
for name in ['data_manifest.json','source_manifest.json']:
 files['provenance/'+name]=(r/name).read_text()
print(json.dumps({'count':len(rows),'files':files}))
'''

def api(endpoint,data=None,method=None):
 cmd=['gh','api',endpoint]
 if method:cmd+=['--method',method]
 if data is not None:cmd+=['--input','-']
 p=subprocess.run(cmd,input=json.dumps(data) if data is not None else None,text=True,capture_output=True,timeout=90)
 if p.returncode:raise RuntimeError(p.stderr[-1200:])
 return json.loads(p.stdout)

def snapshot():
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','galaxy-4090','/home/vipuser/golden-swe-trace-20261103/.venv/bin/python -'],input=SNAPSHOT,text=True,capture_output=True,timeout=60)
 if p.returncode:raise RuntimeError(p.stderr[-1200:])
 return json.loads(p.stdout)

def publish(s):
 files=s['files'];n=s['count']
 files['generator/teacher_bridge.py']=(ROOT/'teacher_bridge.py').read_text()
 files['generator/capacity_retry.py']=(ROOT/'capacity_retry.py').read_text()
 files['generator/publish_remote.py']=pathlib.Path(__file__).read_text()
 files['README.md']=f'''# SWE Golden Trace: GPT-6.1, newly executed R2E-Gym training trajectories

**Published verified unique trajectories: {n} / 1,000 target. Collection is in progress.**

These are SWE repair training demonstrations for Qwen3-4B-Instruct-2507 SFT, separately namespaced from FrozenLake `golden-trace/`. They are not SWE-Bench Verified evaluation results. Teacher: Codex CLI requested model `gpt-6.1-sol`, medium reasoning; the CLI does not independently expose the provider serving-model ID. No model fallback. Data is newly generated through real Docker actions, not copied from public teacher completions.

Training metadata: `zhenghaoxu/R2E-Gym-Lite-with-Difficulty`, revision `87fb43eb4d6c71b1e86292346e45f52b71ea7691`, split `filtered_1_7`. Seed-42 candidate order followed by success filtering. Direct overlap with SWE-Bench Verified is excluded by instance ID, normalized issue text and commit hash; this is not a semantic or pretraining leakage guarantee.

## Acceptance

Each task requires a failing original baseline, a passing reference-source-patch calibration, explicit teacher submission, a nonempty permitted source patch, and independent reapplication/testing in a fresh container. Gold patches and private grading metadata are excluded from teacher prompts and this dataset. The pinned R2E parser compares **all expected test statuses**, including preexisting failures; acceptance does not mean every test is green. `verification.jsonl` reports passing/nonpassing counts and patch/log hashes. Full raw verification logs and teacher invocation records are retained on the remote server.

The Docker adapter uses R2E-Gym source revision `0d94c4eb9431cd195c55a7ea3abd54006c9a1735`; this is an adapter with the upstream status comparison and a nonempty-test check, not an unmodified upstream agent run. Containers have network disabled and no host mounts or Docker socket. Cases whose own reference cannot be certified are recorded as environment rejections, never positive demonstrations.

## Files and usage

- `trajectories/part-*.jsonl`: full accepted episodes, actions, real observations, patch and verification summary; at most 10 trajectories per shard.
- `sft/part-*.jsonl`: observable conversations for supervised training, no hidden chain-of-thought. Concatenate shards once; do not also include `trajectories/` as duplicate SFT examples.
- Action contract: `<function=bash><parameter=command>...</parameter></function>` or `<function=submit></function>`. Train/evaluate with this exact contract or an explicitly tested tool-format conversion. User observations must be masked from the SFT loss.
- `manifest.json`: file hashes, target and accepted counts. Current partial shard may grow; completed shards remain stable.
- `generator/`: generation and relay source. Runner and all durable records live on the remote server; only the authenticated Codex teacher relay and transient response buffers run on the Mac. No account credentials are included. A Mac disconnect pauses teacher delivery; it does not create synthetic observations.

Updates go to **main**. Existing FrozenLake files are unchanged. The 1,000 target is not guaranteed: only unique independently accepted tasks count. No Qwen SFT or PaTR performance improvement has been measured by this collection itself.
'''
 hashes={p:hashlib.sha256(t.encode()).hexdigest() for p,t in files.items()}
 files['manifest.json']=json.dumps({'accepted':n,'target':1000,'complete':n>=1000,'teacher_requested_model':g.MODEL,'dataset_kind':'newly_generated_training_demonstrations','files_sha256':hashes},indent=2)+'\n'
 statepath=ROOT/'publish_status.json'
 old=json.loads(statepath.read_text()) if statepath.exists() else {}
 changed={p:t for p,t in files.items() if old.get('hashes',{}).get(p)!=hashlib.sha256(t.encode()).hexdigest()}
 if not changed:return old
 blobs=[]
 for path,content in changed.items():
  b=api(REPO+'/git/blobs',{'content':base64.b64encode(content.encode()).decode(),'encoding':'base64'})
  blobs.append({'path':PREFIX+'/'+path,'mode':'100644','type':'blob','sha':b['sha']})
 for attempt in range(3):
  head=api(REPO+'/git/ref/heads/main')['object']['sha']
  base=api(REPO+'/git/commits/'+head)['tree']['sha']
  tree=api(REPO+'/git/trees',{'base_tree':base,'tree':blobs})
  commit=api(REPO+'/git/commits',{'message':f'Publish {n} independently verified SWE Golden Traces and remote generation pipeline','tree':tree['sha'],'parents':[head]})
  try:
   api(REPO+'/git/refs/heads/main',{'sha':commit['sha'],'force':False},'PATCH');break
  except RuntimeError:
   if attempt==2:raise
 state={'state':'published','accepted':n,'commit':commit['sha'],'updated':time.time(),'hashes':{p:hashlib.sha256(t.encode()).hexdigest() for p,t in files.items()}}
 g.save(statepath,state)
 # Store publication evidence beside durable remote records, too.
 code="import json,pathlib,sys; p=pathlib.Path('/home/vipuser/golden-swe-trace-20261103/publication_status.json'); t=p.with_suffix('.tmp'); t.write_text(sys.stdin.read()); t.replace(p)"
 import shlex
 subprocess.run(['ssh','galaxy-4090','python3 -c '+shlex.quote(code)],input=json.dumps(state),text=True,check=True,timeout=30)
 print(json.dumps({'published':n,'commit':commit['sha']}),flush=True)
 return state

def main():
 lock=(ROOT/'publish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 once='--once' in sys.argv
 first=True
 while time.time()<g.DEADLINE:
  try:
   s=snapshot();p=ROOT/'publish_status.json';old=json.loads(p.read_text()) if p.exists() else {}
   if first or once or s['count']>=1000 or not old or (s['count']>old.get('accepted',0) and (s['count']-old.get('accepted',0)>=10 or time.time()-old.get('updated',0)>600)):
    publish(s)
    first=False
   if once or s['count']>=1000:return
  except Exception as exc:
   g.save(ROOT/'publish_error.json',{'error':str(exc),'updated':time.time()});print(str(exc),flush=True)
   if once:raise
  time.sleep(45)
if __name__=='__main__':main()
