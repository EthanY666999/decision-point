"""Remote durable queue; teacher-only mailbox served by the authenticated Mac."""
import hashlib,json,pathlib,subprocess,sys,time
import generate as g
ROOT=g.ROOT

def rpc(req,folder,label):
    folder.mkdir(parents=True,exist_ok=True)
    existing=list(folder.glob(label+'.*'))
    if existing:
        import shutil
        archive=folder/'rpc_history'/f'{label}-{time.time_ns()}'
        archive.mkdir(parents=True)
        for previous in existing:
            if previous.is_file():shutil.copy2(previous,archive/previous.name)
    g.save(folder/(label+'.request.json'),req)
    p=subprocess.run([sys.executable,str(ROOT/'remote_env.py')],input=json.dumps(req),text=True,capture_output=True,timeout=2000)
    (folder/(label+'.0.stdout.json')).write_text(p.stdout)
    (folder/(label+'.0.stderr.log')).write_text(p.stderr)
    if not p.stdout.strip(): raise RuntimeError('Environment process returned no result; inspect saved state')
    r=json.loads(p.stdout)
    if not r.get('ok'): raise RuntimeError(r.get('error','Environment failed'))
    return r['result']

def teacher(messages,folder,turn):
    path=folder/f'teacher-{turn:03d}'
    meta_path=path.with_suffix('.meta.json');output=path.with_suffix('.response.json')
    request={'instruction':'Act as the policy for the conversation below. Return only JSON with action (bash or submit), command (empty for submit), and summary (one brief sentence or empty). Do not use Codex tools, run local commands, access files or delegate. The separate remote environment executes your proposed command and supplies the next observation. Never invent observations.', 'conversation':messages}
    fingerprint=g.sha(json.dumps(request,ensure_ascii=False).encode())
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        assert meta['prompt_sha256']==fingerprint
        assert meta['response_sha256']==g.sha(output.read_bytes())
        return json.loads(output.read_text()),meta
    pending={'index':int(folder.name),'turn':turn,'messages':messages,'prompt_sha256':fingerprint}
    g.save(ROOT/'teacher_pending.json',pending)
    ready=path.with_suffix('.ready.json')
    g.event('waiting_for_teacher',index=int(folder.name),turn=turn)
    while not ready.exists():
        if time.time()>g.DEADLINE: raise RuntimeError('Rental cutoff reached while waiting for teacher')
        if (ROOT/'STOP').exists(): raise RuntimeError('STOP file found while waiting for teacher')
        time.sleep(2)
    ack=json.loads(ready.read_text())
    if ack.get('error'):raise RuntimeError('Teacher bridge failed: '+ack['error'])
    meta=json.loads(meta_path.read_text())
    assert meta['prompt_sha256']==fingerprint
    assert meta['response_sha256']==g.sha(output.read_bytes())
    (ROOT/'teacher_pending.json').unlink(missing_ok=True)
    return json.loads(output.read_text()),meta

g.rpc=rpc;g.teacher=teacher
if __name__=='__main__':g.main()
