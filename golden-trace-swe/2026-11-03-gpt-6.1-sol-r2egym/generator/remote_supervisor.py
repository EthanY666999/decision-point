"""Resume known-safe image-download failures while preserving every worker exit."""
import fcntl,json,os,pathlib,subprocess,sys,time
import generate as g
from recovery_utils import safe_image_restart,safe_preparing_resume
ROOT=g.ROOT

def main():
    lock=(ROOT/'remote_supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    retries=0
    inspected_initial_resume="--resume-inspected" in sys.argv
    while time.time()<g.DEADLINE and not (ROOT/'STOP').exists():
        # Do not launch against an already-running queue.
        with (ROOT/'run.lock').open('a') as runlock:
            try:fcntl.flock(runlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Existing worker owns run.lock; refusing duplicate launch')
        status=json.loads((ROOT/'status.json').read_text())
        if not (safe_image_restart(ROOT,status) or (inspected_initial_resume and retries==0 and safe_preparing_resume(ROOT))):
            g.save(ROOT/'supervisor_status.json',{'state':'attention_required','reason':'Exit is not a certified pre-container image failure','worker_status':status,'pid':os.getpid(),'updated':time.time()});return
        archive=ROOT/'recovery_history';archive.mkdir(exist_ok=True)
        g.save(archive/f'worker-exit-{time.time_ns()}.json',status)
        if retries:
            until=min(time.time()+min(60*2**min(retries-1,3),600),g.DEADLINE)
            g.save(ROOT/'supervisor_status.json',{'state':'image_retry_wait','retry_at':until,'retries':retries,'pid':os.getpid(),'updated':time.time()})
            while time.time()<until:
                if (ROOT/'STOP').exists():return
                time.sleep(max(0,min(15,until-time.time())))
        if time.time()>=g.DEADLINE:return
        with (ROOT/'remote_runner.log').open('a') as out:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'remote_runner.py'),'--target','1000'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            g.save(ROOT/'supervisor_status.json',{'state':'worker_running','worker_pid':p.pid,'pid':os.getpid(),'restarts':retries+1,'updated':time.time()})
            code=p.wait()
        status=json.loads((ROOT/'status.json').read_text())
        g.save(archive/f'worker-exit-{time.time_ns()}.json',{'exit_code':code,'status':status})
        if code==0:
            g.save(ROOT/'supervisor_status.json',{'state':'worker_completed','worker_status':status,'pid':os.getpid(),'updated':time.time()});return
        retries+=1
    g.save(ROOT/'supervisor_status.json',{'state':'stopped_at_cutoff_or_stop_file','pid':os.getpid(),'updated':time.time()})
if __name__=='__main__':main()
