"""Isolated R2E training task execution. JSON in/out over authenticated SSH.

Uses the pinned upstream ParsedCommit and log parser, plus its exact-match
reward rule with an additional nonempty-test requirement. It intentionally does
not import upstream's API teacher, Kubernetes or model-training dependencies.
"""
import contextlib
import fcntl
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import sys
import tarfile
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'upstream_src'))
import docker
import pyarrow.parquet as pq
from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.repo_analysis.execution_log_parser import parse_log_fn, decolor_dict_keys

STATE = ROOT/'environment_states'
STATE.mkdir(exist_ok=True)
LABEL = 'ethan.golden-swe.20261103'
PATH = '/testbed/.venv/bin:/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
LIMIT = 24000

def sha(b):
    return hashlib.sha256(b).hexdigest()

def save(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)

def row_for(index):
    c = json.loads((ROOT/'candidates.json').read_text())[index]
    path = ROOT/'data/R2E-Gym-Lite-with-Difficulty'/c['shard']
    row = pq.read_table(path).slice(c['row_index'],1).to_pylist()[0]
    assert row['instance_id'] == c['instance_id']
    return row

def check_space():
    free = shutil.disk_usage('/var/lib').free
    if free < 15 * 1024**3:
        raise RuntimeError('Storage reserve reached; stop before pulling another image')

def run(c, command, seconds=120):
    t=time.monotonic()
    r=c.exec_run(['timeout','--signal=TERM','--kill-after=5',str(seconds),
                  '/bin/bash','-c',command],workdir='/testbed',environment={'PATH':PATH})
    return {'output':r.output.decode('utf-8',errors='replace'),
            'exit_code':r.exit_code,'seconds':round(time.monotonic()-t,3)}

def checked(c, command, seconds=120):
    r=run(c,command,seconds)
    if r['exit_code'] != 0:
        raise RuntimeError('Environment setup failed: '+command[:80]+' '+r['output'][-2000:])
    return r

def put(c, path, data):
    if isinstance(data,str): data=data.encode()
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as t:
        i=tarfile.TarInfo(pathlib.PurePosixPath(path).name)
        i.size=len(data); i.mode=0o600
        t.addfile(i,io.BytesIO(data))
    assert c.put_archive(str(pathlib.PurePosixPath(path).parent),buf.getvalue())

def create(client, image):
    return client.containers.run(image,['sleep','infinity'],detach=True,
        name='golden-swe-'+uuid.uuid4().hex[:16],labels={LABEL:'owned'},
        network_mode='none',mem_limit='8g',nano_cpus=6_000_000_000,pids_limit=512,
        cap_drop=['ALL'],security_opt=['no-new-privileges:true'],
        log_config=docker.types.LogConfig(type='local',config={'max-size':'10m','max-file':'3'}))

def setup(c):
    # These are disposable containers only. Never operate on host experiment files.
    checked(c,'test -d /testbed && test -d /testbed/.git')
    info=checked(c,'git rev-parse HEAD; git status --porcelain')['output']
    # Versioneer/setuptools_scm repos (numpy, pandas, ...) derive __version__ from the nearest
    # git tag. Removing history makes it '0+untagged' and breaks test collection, so remember
    # the nearest release tag name (a version label only, no answer content) and restore it.
    described=run(c,'git describe --tags --abbrev=0 2>/dev/null')
    version_tag=described['output'].strip().splitlines()[-1].strip() if described['exit_code']==0 and described['output'].strip() else ''
    if not re.fullmatch(r'[A-Za-z0-9._+\-]{1,100}',version_tag): version_tag=''
    # Remove answer-bearing history and legacy image metadata before any teacher action.
    checked(c,"find /testbed -name .git -type d -prune -exec rm -rf {} +")
    checked(c,"find /testbed /root -type f \\( -name parsed_commit.json -o -name syn_issue.json -o -name expected_test_output.json -o -name execution_result.json -o -name modified_files.json -o -name modified_entities.json -o -name '*.patch' \\) -delete")
    # The fixed script and R2E tests are part of the image. Keep them unchanged.
    checked(c,'mkdir -p /root/.local/bin; ln -sfn /testbed/.venv /root/.venv')
    checked(c,'if test -d /r2e_tests && ! test -e /testbed/r2e_tests; then ln -s /r2e_tests /testbed/r2e_tests; elif test -d /root/r2e_tests && ! test -e /testbed/r2e_tests; then ln -s /root/r2e_tests /testbed/r2e_tests; fi')
    test_path=checked(c,"for p in /testbed/run_tests.sh /root/run_tests.sh /run_tests.sh; do if test -f \"$p\"; then echo \"$p\"; break; fi; done")['output'].strip()
    if not test_path: raise RuntimeError('Missing fixed image test script')
    checked(c,'git init -q; git config user.email golden-trace@localhost; git config user.name GoldenTrace')
    checked(c,"printf '\n.venv/\n__pycache__/\n*.pyc\nr2e_tests/\nrun_tests.sh\n' >> .git/info/exclude; git add -A; git commit -qm 'Isolated task baseline'",180)
    if version_tag: checked(c,'git tag '+version_tag)
    return {'image_git_state_before_sanitization':info,'test_command':'bash '+test_path,
            'restored_version_tag':version_tag,
            'baseline_commit':checked(c,'git rev-parse HEAD')['output'].strip()}

def grade(c, row, setup_info, outdir, tag):
    result=run(c,setup_info['test_command'],600)
    parsed=decolor_dict_keys(parse_log_fn(row['repo_name'])(result['output']))
    expected=decolor_dict_keys(json.loads(row['expected_output_json']))
    parsed={k.split(' - ')[0]:parsed[k] for k in sorted(parsed)}
    expected={k.split(' - ')[0]:expected[k] for k in sorted(expected)}
    # Upstream allows empty == empty; refuse that vacuous success.
    passed=bool(expected) and parsed==expected and result['exit_code'] not in (124,137,143)
    log=outdir/(tag+'.log'); log.write_text(result['output'])
    report={'passed':passed,'exit_code':result['exit_code'],'seconds':result['seconds'],
            'expected_count':len(expected),'parsed_count':len(parsed),
            'matching_count':sum(parsed.get(k)==v for k,v in expected.items()),
            'expected_passing_tests':sum(v=='PASSED' for v in expected.values()),
            'expected_nonpassing_tests':sum(v!='PASSED' for v in expected.values()),
            'parsed':parsed,'expected':expected,'log_sha256':sha(log.read_bytes())}
    save(outdir/(tag+'.json'),report)
    return report

def get_state(token):
    if not re.fullmatch(r'[a-f0-9]{32}',token): raise ValueError('Invalid environment token')
    d=STATE/token
    return d,json.loads((d/'state.json').read_text())

def safe_action_paths(files):
    for f in files:
        p=pathlib.PurePosixPath(f)
        if p.is_absolute() or '..' in p.parts: return False
        if any(x.lower() in {'tests','test','r2e_tests','.git','.venv'} for x in p.parts): return False
        if p.name.startswith('test_') or p.name.endswith('_test.py'): return False
        if p.name in {'conftest.py','run_tests.sh','pytest.ini','tox.ini'}: return False
    return True

def pull(client,image):
    check_space()
    try: return client.images.get(image),False
    except docker.errors.ImageNotFound: pass
    last=None
    for _ in range(3):
        try: return client.images.pull(image),True
        except Exception as e: last=e
    raise last

def begin(client, req):
    row=row_for(req['index'])
    token=req['token']
    if not re.fullmatch(r'[a-f0-9]{32}',token): raise ValueError('Invalid token')
    out=STATE/token
    if (out/'state.json').exists():
        old=json.loads((out/'state.json').read_text())
        if old['phase']=='teacher':
            return {'token':token,'instance_id':row['instance_id'],
                    'problem_statement':(re.search(r'\[ISSUE\](.*?)\[/ISSUE\]',row['problem_statement'],re.S).group(1).strip() if re.search(r'\[ISSUE\](.*?)\[/ISSUE\]',row['problem_statement'],re.S) else row['problem_statement']),'repo':row['repo_name'],
                    'calibration':old['calibration']}
        raise RuntimeError('Existing environment requires inspection: '+old['phase'])
    out.mkdir(exist_ok=True)
    save(out/'task_private.json',row)
    image,owned=pull(client,row['docker_image'])
    state={'token':token,'index':req['index'],'instance_id':row['instance_id'],
           'image':row['docker_image'],'image_id':image.id,'digests':image.attrs.get('RepoDigests',[]),
           'downloaded_by_this_run':owned,'phase':'calibrating','created':time.time()}
    save(out/'state.json',state)
    c=create(client,image.id)
    try:
        info=setup(c); state['setup']=info
        base=grade(c,row,info,out,'baseline')
        if base['passed']: raise RuntimeError('Original task already passes; cannot certify a repair')
        patch=ParsedCommit(**json.loads(row['parsed_commit_content'])).get_patch(test_file=False,non_test_file=True)
        if not patch.strip(): raise RuntimeError('Empty reference source patch')
        put(c,'/tmp/reference.patch',patch)
        checked(c,'git apply --whitespace=nowarn /tmp/reference.patch')
        gold=grade(c,row,info,out,'reference')
        if not gold['passed']: raise RuntimeError('Reference patch does not pass; task environment rejected')
        state['calibration']={'baseline_passed':False,'reference_passed':True,
                              'expected_tests':gold['expected_count'],
                              'expected_passing_tests':gold['expected_passing_tests'],
                              'expected_nonpassing_tests':gold['expected_nonpassing_tests'],
                              'failed_to_pass_tests':sum(v=='PASSED' and base['parsed'].get(k)!='PASSED' for k,v in gold['expected'].items())}
    except Exception as e:
        state.update(phase='environment_rejected',error=str(e));save(out/'state.json',state)
        raise
    finally:
        c.remove(force=True)
    c=create(client,image.id)
    try: state['setup']=setup(c)
    except BaseException:
        c.remove(force=True);raise
    state.update(container=c.id,phase='teacher',step=0)
    save(out/'state.json',state)
    problem=row['problem_statement']
    m=re.search(r'\[ISSUE\](.*?)\[/ISSUE\]',problem,re.S)
    if m: problem=m.group(1).strip()
    return {'token':token,'instance_id':row['instance_id'],'problem_statement':problem,
            'repo':row['repo_name'],'calibration':state['calibration'],
            'available_tools':checked(c,'for t in python python3 grep find sed rg git; do command -v "$t" || true; done')['output']}

def act(client, req):
    out,state=get_state(req['token'])
    assert state['phase']=='teacher'
    step=int(req['step']); resultfile=out/f'action-{step:03d}.json'
    command=req['command']; fingerprint=sha(command.encode())
    # Network retries must not execute a mutation twice.
    if resultfile.exists():
        saved=json.loads(resultfile.read_text());assert saved['command_sha256']==fingerprint
        return saved['observation']
    assert step==state['step']+1
    running=out/f'action-{step:03d}.running'
    if running.exists(): raise RuntimeError('Ambiguous interrupted action; manual recovery needed')
    running.write_text(fingerprint)
    c=client.containers.get(state['container'])
    r=run(c,command)
    (out/f'action-{step:03d}.full.log').write_text(r['output'])
    output=r['output']
    if len(output)>LIMIT:
        output=output[:LIMIT//2]+'\n[output clipped; complete output retained in audit log]\n'+output[-LIMIT//2:]
    obs={**r,'output':output}
    save(resultfile,{'command':command,'command_sha256':fingerprint,'observation':obs})
    state['step']=step;save(out/'state.json',state);running.unlink()
    return obs

def finish(client,req):
    out,state=get_state(req['token'])
    if state['phase']=='complete': return state['result']
    assert state['phase']=='teacher'
    row=json.loads((out/'task_private.json').read_text())
    c=client.containers.get(state['container'])
    checked(c,'git add -A')
    files=checked(c,'git diff --cached --name-only -z')['output'].strip('\0').split('\0')
    patch=checked(c,'git diff --cached --binary')['output']
    (out/'teacher.patch').write_text(patch)
    result={'accepted':False,'instance_id':state['instance_id'],'token':state['token'],
            'patch_sha256':sha(patch.encode()),'files':files,'image_id':state['image_id'],
            'image_digests':state['digests'],'calibration':state['calibration']}
    if not patch.strip() or not safe_action_paths(files):
        result['reason']='empty patch or forbidden test/configuration path'
    else:
        # Grading in an independent, fresh image prevents modified tests and cached output cheating.
        verify=create(client,state['image_id'])
        try:
            info=setup(verify)
            put(verify,'/tmp/teacher.patch',patch)
            checked(verify,'git apply --whitespace=nowarn /tmp/teacher.patch')
            g=grade(verify,row,info,out,'independent_verification')
            result.update(accepted=g['passed'],reason='verified' if g['passed'] else 'tests_failed',
                          expected_tests=g['expected_count'],matching_tests=g['matching_count'],
                          expected_passing_tests=g['expected_passing_tests'],
                          expected_nonpassing_tests=g['expected_nonpassing_tests'],
                          acceptance_semantics='Matches every upstream expected test status, including preexisting failures; not necessarily an all-green test suite')
        finally: verify.remove(force=True)
    c.remove(force=True)
    result['patch']=patch
    state.update(phase='complete',result=result,completed=time.time());save(out/'state.json',state)
    # Release only an image downloaded for this task, never prune shared/user images.
    if state['downloaded_by_this_run'] and not client.containers.list(all=True,filters={'ancestor':state['image_id']}):
        try:
            client.images.remove(state['image'],force=False,noprune=False)
            state['owned_image_released']=True;save(out/'state.json',state)
        except docker.errors.APIError as e:
            state['image_release_note']=str(e);save(out/'state.json',state)
    return result

def main():
    req=json.load(sys.stdin)
    if req.get('token'):
        if not re.fullmatch(r'[a-f0-9]{32}',req['token']): raise ValueError('Invalid token')
        lock=(STATE/(req['token']+'.lock')).open('w')
        fcntl.flock(lock,fcntl.LOCK_EX)
    client=docker.from_env(timeout=900)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if req['op']=='begin': result=begin(client,req)
            elif req['op']=='act': result=act(client,req)
            elif req['op']=='finish': result=finish(client,req)
            elif req['op']=='status':
                result={'states':[json.loads(p.read_text()) for p in STATE.glob('*/state.json')]}
            else: raise ValueError('Unknown operation')
        print(json.dumps({'ok':True,'result':result},ensure_ascii=False))
    except Exception as e:
        print(json.dumps({'ok':False,'error':type(e).__name__+': '+str(e)},ensure_ascii=False))
        sys.exit(1)
    finally: client.close()

if __name__=='__main__': main()
