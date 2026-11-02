"""Two-phase, receipt-backed removal of locally SHA-verified checkpoint replicas."""
import argparse, fcntl, hashlib, json, os, shutil, subprocess, time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','clean']);p.add_argument('--github-commit');a=p.parse_args()
ROOT=Path('/home/vipuser/golden-cert-20261102-r1/code/artifacts/disk_20261103')
m=json.loads((ROOT/'manifest.json').read_text())
assert m['complete'] and len(m['files'])==7
lock=Path('/home/vipuser/golden-controls-20261031/code/runs/control_queue/supervisor.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

def save(name,obj):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');os.replace(t,p)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def users(p):
    actual=str(p.resolve());found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    if os.path.realpath(fd)==actual:found.append(proc.name)
                except OSError:pass
            if actual in (proc/'maps').read_text():found.append(proc.name)
        except (OSError,PermissionError):pass
    return sorted(set(found))
def protected():
    expected=json.loads((ROOT/'protected_inputs_sha256.json').read_text())
    root=Path('/home/vipuser/golden-controls-20261031/code')
    actual={k:digest(root/k) for k in expected}
    assert actual==expected,'Protected files changed'
    return actual
def disks():return {p:dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in ['/','/data']}

gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu, 'GPU active: '+gpu
checks=protected()
if a.mode=='audit':
    audit={'time':time.time(),'before':disks(),'protected':checks,'files':[]}
    for e in m['files']:
        s=Path(e['source']);r=Path(e['run']);state=json.loads((r/'status.json').read_text())
        assert s.name=='checkpoint.pt' and 'extended800' not in str(s)
        assert state['status']=='complete' and state['step']==state['total_steps']==400
        target=s.resolve(strict=True);st=target.stat()
        assert target.is_file() and st.st_size==e['bytes'] and digest(target)==e['sha256']
        assert not users(target)
        audit['files'].append(dict(e,target=str(target),source_is_symlink=s.is_symlink(),inode=st.st_ino,mtime_ns=st.st_mtime_ns,remote_sha256=e['sha256'],users=[]))
        print('remote_verified',e['label'],flush=True)
    audit['complete']=True;save('pre_cleanup.json',audit)
else:
    assert a.github_commit and len(a.github_commit)>=7
    audit=json.loads((ROOT/'pre_cleanup.json').read_text());assert audit['complete'] and len(audit['files'])==7
    receipt={'started':time.time(),'github_pre_sync':a.github_commit,'before':disks(),'files':[],'protected_before':checks}
    save('receipt.json',receipt)
    # Recheck every candidate before the first deletion.
    for e in audit['files']:
        s=Path(e['source']);t=Path(e['target']);st=t.stat()
        assert s.resolve(strict=True)==t and st.st_size==e['bytes'] and st.st_ino==e['inode'] and st.st_mtime_ns==e['mtime_ns'] and not users(t)
        state=json.loads((Path(e['run'])/'status.json').read_text());assert state['status']=='complete' and state['step']==400
    for e in audit['files']:
        s=Path(e['source']);t=Path(e['target']);assert not users(t)
        t.unlink()
        if s.is_symlink():s.unlink()
        assert not s.exists() and not s.is_symlink()
        receipt['files'].append(dict(e,deleted_unix=time.time()));save('receipt.json',receipt)
        print('removed_redundant_checkpoint',e['label'],flush=True)
    receipt.update(status='complete_verified',finished=time.time(),after=disks(),protected_after=protected())
    save('receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'before':receipt['before'],'after':receipt['after']}),flush=True)
