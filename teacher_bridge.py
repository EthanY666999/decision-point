"""Teacher calls only on Mac; all trajectory data and full call evidence stay remote."""
import datetime,fcntl,json,os,re,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODEL='gpt-6.1-sol'
SSH=['ssh','-S','/tmp/ucf-q35-coord.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','ucf-2b-80g']
REMOTE='/data/venvs/qwen35/bin/python /home/vipuser/golden-flh-trace-20261108/remote_dataset.py '
def remote(command,packet=None):
 r=subprocess.run(SSH+['nice -n 10 '+REMOTE+command],input=json.dumps(packet) if packet is not None else None,capture_output=True,text=True,timeout=180)
 if r.returncode:raise RuntimeError(r.stderr[-2000:])
 return json.loads(r.stdout)
def report(state,**kw):
 d=dict(state=state,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**kw)
 tmp=ROOT/'bridge_status.tmp';tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(ROOT/'bridge_status.json');print(json.dumps(d,ensure_ascii=False),flush=True)
def teacher(req):
 cmd=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','--model',MODEL,'-c','model_reasoning_effort="medium"','--disable','shell_tool','--disable','multi_agent','--disable','apps','--disable','plugins','--sandbox','read-only','-C',str(ROOT),'--json','--output-schema',str(ROOT/'response_schema.json'),'-']
 env=os.environ.copy();proxy=subprocess.check_output(['scutil','--proxy'],text=True)
 host=re.search(r'HTTPSProxy\s*:\s*(\S+)',proxy);port=re.search(r'HTTPSPort\s*:\s*(\d+)',proxy)
 if host and port and re.search(r'HTTPSEnable\s*:\s*1',proxy):env['HTTPS_PROXY']=env['HTTP_PROXY']=f'http://{host.group(1)}:{port.group(1)}'
 started=time.monotonic()
 result=subprocess.run(cmd,input=json.dumps(req),capture_output=True,text=True,env=env,timeout=600)
 return dict(request=req,requested_model=MODEL,call_id=uuid.uuid4().hex,argv=cmd,stdout=result.stdout,stderr=result.stderr,returncode=result.returncode,seconds=time.monotonic()-started)
def main():
 lock=(ROOT/'bridge.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status=remote('status');report('connected',remote=status)
 if status.get('state')=='pilot_quality_failed':raise RuntimeError('pilot failed; review before batch continuation')
 while True:
  req=remote('request')
  if req.get('complete'):
   summary=remote('verify');report('complete',summary=summary);return
  report('teacher_running',batch=req['batch_id'],games=len(req['games']),last_remote=status)
  packet=teacher(req)
  # A connection failure must not trigger another paid model call. Retry delivery only.
  for attempt in range(3):
   try:status=remote('ingest',packet);break
   except Exception:
    if attempt==2:raise
    time.sleep(3)
  report('batch_verified',remote=status,teacher_seconds=packet['seconds'])
  if status.get('state')=='pilot_quality_failed':return
  if '--pilot' in sys.argv:return
if __name__=='__main__':
 try:main()
 except BaseException as ex:report('stopped_error',error=str(ex));raise
