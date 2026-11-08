"""One explicitly authorized S05 start; no implicit retry or source edits."""
import fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/vipuser/decision-point-q35-hard/code')
NEW=Path('/home/vipuser/decision-point-q35-loader/code')
STATE=Path('/data/runs/queue_q35_hard');RUNS=Path('/data/runs')
HERE=STATE/'S05_startup_repair_20261109';OUT=RUNS/'hard_queue/S05_ucf_seed42'
PY='/data/venvs/qwen35/bin/python'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from q35_run_receipt import read,atomic,sha,verify
from q35_p0_supervisor import relocate_verified_model
JOURNAL=STATE/'attempts/S05_ordinary_recovery_20261109.json'
def state(**kw):atomic(HERE/'recovery_controller_status.json',dict(unix=time.time(),controller_pid=os.getpid(),**kw))
def main():
    own=(STATE/'supervisor.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=(RUNS/'gpu80.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    auth=read(HERE/'authorization.json');assert auth['extra_launches']==1 and not auth['preflight_failures_consume_training_recovery']
    assert not JOURNAL.exists(),'Single authorized launch journal already exists; inspect, never relaunch'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
    assert shutil.disk_usage(RUNS).free>=55*2**30
    assert read(RUNS/'hard_queue/S04_tree_random_seed42/verification.json')['passed']
    assert read(RUNS/'hard_queue/S02_SFT_seed42/verification.json')['passed']
    assert read(RUNS/'stage0/sokoban_acceptance/acceptance.json')['passed']
    preflight=subprocess.run([PY,str(HERE/'launch_same_source.py')],cwd=ROOT,capture_output=True,text=True,check=True)
    atomic(HERE/'preflight_before_launch.json',dict(passed=True,stdout=preflight.stdout,checked_unix=time.time()))
    cfg=read(STATE/'configs/S05.json');entry=next(e for e in read(STATE/'plan.json')['entries'] if e['id']=='S05');assert cfg==entry['config']
    first_attempt=read(STATE/'attempts/S05_authorized_start_20261109.json')
    assert first_attempt['returncode']==1 and first_attempt['training_attempt']==1 and not first_attempt['ordinary_recovery_used']
    assert not (OUT/'checkpoint.pt').exists(),'Checkpoint present: use verified resume instead'
    assert not (OUT/'train_metrics.jsonl').exists(),'Unexpected completed update; inspect before restart'
    archive=OUT.with_name(OUT.name+'_training_attempt1_20261109')
    assert not archive.exists()
    assert sha(Path('/data/venvs/qwen35/lib/python3.12/site-packages/vllm/v1/worker/gpu_input_batch.py'))==read(HERE/'vllm_mask_repair/receipt.json')['after_sha256']
    original=STATE/'attempts/hard_queue__S05_ucf_seed42.json';assert len(read(original))==2
    evidence={str(x):sha(x) for x in [original,OUT/'needs_manual.json',STATE/'handoff_status.json']}
    atomic(HERE/'preserved_training_failure_evidence.json',dict(record_sha256={str(x.relative_to(OUT)):sha(x) for x in OUT.rglob('*') if x.is_file()},training_attempts_before_this=1,archive=str(archive)))
    OUT.rename(archive);OUT.mkdir()
    atomic(OUT/'environment_repair.json',read(HERE/'vllm_mask_repair/receipt.json'))
    atomic(OUT/'launch_config.json',cfg)
    atomic(OUT/'hardware.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),code_commit=auth['source_commit'],model_family='Qwen3.5',config_difference=entry['config_difference'],source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST_q35.json'),external_launch_repair=str(HERE/'preflight_before_launch.json')))
    command=[PY,'-u',str(HERE/'launch_same_source.py'),'--launch']
    rec=dict(command=command,resume=False,started=time.time(),authorization_sha256=sha(HERE/'authorization.json'),source_commit=auth['source_commit'],config_sha256=sha(STATE/'configs/S05.json'),original_preflight_attempts_sha256=sha(original),training_attempt=2,ordinary_recovery_used=True)
    with JOURNAL.open('x') as f:json.dump(rec,f,indent=2)
    atomic(STATE/'status.json',dict(status='running',experiment='S05',pid=os.getpid(),authorization='S05_extra_start_20261109'))
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1')
    with (HERE/'recovery_training.log').open('a') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(gpu.fileno(),))
        rec['pid']=child.pid;atomic(JOURNAL,rec);state(status='training',training_pid=child.pid)
        rec['returncode']=child.wait();rec['ended']=time.time();atomic(JOURNAL,rec)
    assert sha(original)==rec['original_preflight_attempts_sha256']
    if rec['returncode']!=0:
        state(status='needs_manual_two_training_failures',returncode=rec['returncode'],ordinary_recovery_available=False)
        atomic(OUT/'needs_manual.json',dict(status='待人工',reason='First actual training attempt and the one ordinary recovery failed; all evidence retained'))
        return
    assert read(OUT/'status.json')['status']=='complete'
    state(status='verifying_final_records')
    verify(OUT,prune_checkpoint=True);relocate_verified_model(OUT)
    state(status='S05_verified_waiting_handoff')
    for f in [gpu,own]:fcntl.flock(f,fcntl.LOCK_UN);f.close()
    old=STATE/'handoff_status.json';saved=HERE/'handoff_status_before_success.json'
    if old.exists():assert not saved.exists();shutil.copy2(old,saved)
    env=dict(os.environ,PYTHONPATH=str(NEW))
    subprocess.run([PY,'-u',str(NEW/'scripts/q35_after_s05_handoff.py')],cwd=NEW,env=env,check=True)
    state(status='completed_and_handed_off')
if __name__=='__main__':
    try:main()
    except Exception as exc:
        state(status='needs_manual',error=repr(exc),traceback=traceback.format_exc());raise
