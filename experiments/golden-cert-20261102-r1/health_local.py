"""Read-only queue, process identity and disk snapshot; never opens active weights."""
from pathlib import Path
import subprocess,sys
stage=sys.argv[1];stamp=sys.argv[2];assert stage in ('uct','uniform')
s='''from pathlib import Path
import json,time,subprocess,shutil
r=Path('/home/vipuser/golden-cert-20261102-r1/code');q=json.loads((r/'runs/queue/status.json').read_text());pid=q.get('pid');f=r/'runs/g2/SELECTOR/code/runs/formal/patr_seed42';cp=f/'checkpoint.pt'
p=Path(f'/proc/{pid}/stat');ident=int(p.read_text().split(') ',1)[1].split()[19]) if p.exists() else None
print(json.dumps(dict(time=time.time(),queue=q,start_ticks_live=ident,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'],text=True),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),disk={x:shutil.disk_usage(x).free for x in ['/','/data']},status=json.loads((f/'status.json').read_text()) if (f/'status.json').exists() else None,progress=json.loads((f/'progress.json').read_text()) if (f/'progress.json').exists() else None,checkpoint_stat=dict(size=cp.stat().st_size,mtime=cp.stat().st_mtime) if cp.exists() else None)))
'''.replace('SELECTOR',stage)
for attempt in range(3):
 p=subprocess.run(['ssh','-o','IPQoS=none','-o','ConnectTimeout=15','galaxy-4090','/home/vipuser/patr-reproduction/.venv/bin/python','-'],input=s,text=True,capture_output=True)
 if p.returncode==0:break
 print(p.stderr,file=sys.stderr)
p.check_returncode();d=Path(__file__).resolve().parents[2]/'output/golden-cert-20261102-r1'/('monitor_'+stamp);d.mkdir(exist_ok=True);(d/'health.json').write_text(p.stdout);print(p.stdout)
