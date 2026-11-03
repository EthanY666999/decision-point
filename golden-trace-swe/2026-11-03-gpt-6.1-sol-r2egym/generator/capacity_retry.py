"""Retry only explicit service-capacity failures; retain failed calls remotely."""
import json,pathlib,time,os,subprocess

def is_capacity_failure(folder,turn):
    p=pathlib.Path(folder)/f'teacher-{turn:03d}.events.jsonl'
    if not p.exists():return False
    try:events=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    except (ValueError,OSError):return False
    if any(e.get('type')=='turn.completed' for e in events):return False
    errors=[e.get('message','') if e.get('type')=='error' else e.get('error',{}).get('message','') for e in events if e.get('type') in ('error','turn.failed')]
    return bool(errors) and all('selected model is at capacity' in e.lower() for e in errors)

def archive_failure(g,folder,turn):
    # Move only this failed call after creating a dated folder, then sync it before retry.
    folder=pathlib.Path(folder)
    dest=folder/'failed_attempts'/f'turn-{turn:03d}-{time.time_ns()}'
    dest.mkdir(parents=True)
    for p in folder.glob(f'teacher-{turn:03d}.*'):p.rename(dest/p.name)
    p=subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',str(folder)+'/', 'galaxy-4090:'+g.REMOTE+f'/episodes/{folder.name}/'],capture_output=True,text=True,timeout=90)
    if p.returncode:raise RuntimeError('Failed-call evidence sync failed: '+p.stderr[-500:])

def call_with_retry(g,messages,folder,turn):
    if is_capacity_failure(folder,turn):archive_failure(g,folder,turn)
    retries=0
    while time.time()<g.DEADLINE:
        g.save(g.ROOT/'bridge_status.json',{'state':'teacher_call','index':int(folder.name),'turn':turn,'pid':os.getpid(),'capacity_retries':retries,'updated':time.time()})
        try:return g.teacher(messages,folder,turn)
        except AssertionError:
            if not is_capacity_failure(folder,turn):raise
            archive_failure(g,folder,turn)
            retries+=1
            delay=min(120*2**min(retries-1,3),600)
            until=min(time.time()+delay,g.DEADLINE)
            g.save(g.ROOT/'bridge_status.json',{'state':'capacity_retry','index':int(folder.name),'turn':turn,'pid':os.getpid(),'model':g.MODEL,'capacity_retries':retries,'retry_at':until,'updated':time.time()})
            print(f'Teacher capacity unavailable; same-model retry in {delay}s',flush=True)
            while time.time()<until:time.sleep(min(30,until-time.time()))
    raise RuntimeError('Rental computation cutoff reached during capacity retry')
