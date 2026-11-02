"""One-time CPU-only handoff; refuses to signal any training or GPU PID."""
import json,os,signal,subprocess,time
from pathlib import Path
from interleave_queue import identity,owners,read,save,ROOT,B3,PY,OUT,ready

def main():
 assert ready();state=read(B3/'runs/queue/status.json')
 assert state['stage']=='patr_400' and state['status']=='running'
 trainer=state['pid'];supervisor=state['supervisor_pid'];assert trainer!=supervisor
 assert owners()=={trainer}
 def cmd(pid):return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 assert 'scripts/b3_queue.py' in cmd(supervisor) and 'run_b3.py --method patr' in cmd(trainer)
 ticks=identity(trainer);st=identity(supervisor);assert ticks and st
 assert int((Path('/proc')/str(trainer)/'stat').read_text().rsplit(')',1)[1].split()[1])==supervisor
 fds=list((Path('/proc')/str(trainer)/'fd').iterdir())
 assert any(os.path.realpath(fd)=='/home/vipuser/golden-controls-20261031/code/runs/control_queue/supervisor.lock' for fd in fds),'Trainer must retain shared GPU lock after CPU supervisor exits'
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'handover.json').exists(),'No duplicate handoff'
 hand=dict(time=time.time(),trainer_pid=trainer,trainer_start_ticks=ticks,old_supervisor_pid=supervisor,old_supervisor_start_ticks=st,old_queue_status=state,policy='Signal ONLY old CPU supervisor; adopted trainer untouched; new CPU waiter acquires lease only after natural exit')
 save(OUT/'handover.json',hand)
 with (OUT/'supervisor.log').open('a') as f:
  waiter=subprocess.Popen([PY,'-u','scripts/interleave_queue.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 for _ in range(50):
  if (OUT/'status.json').exists() and read(OUT/'status.json').get('status')=='waiting_live_B3_PaTR':break
  assert waiter.poll() is None;time.sleep(.1)
 else:raise RuntimeError('CPU waiter did not become ready; original supervisor remains untouched')
 assert identity(supervisor)==st and identity(trainer)==ticks and owners()=={trainer}
 os.kill(supervisor,signal.SIGTERM)
 for _ in range(50):
  if identity(supervisor) is None:break
  time.sleep(.1)
 assert identity(supervisor) is None and identity(trainer)==ticks and owners()=={trainer}
 hand.update(new_waiter_pid=waiter.pid,new_waiter_start_ticks=identity(waiter.pid),finished=time.time(),trainer_preserved=True,shared_lease_retained=True)
 save(OUT/'handover.json',hand)
 save(B3/'runs/queue/status.json',dict(status='running',stage='patr_400',pid=trainer,supervisor_pid=waiter.pid,supervisor_kind='CERT_CPU_handover_waiter',handover=str(OUT/'handover.json'),started_unix=state['started_unix'],command=state['command'],next='PaTR supp500 then CERT G1 then original Random400; do not restart old supervisor while live child'))
 print(json.dumps(hand))
if __name__=='__main__':main()
