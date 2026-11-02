"""G2 entrypoint: frozen optimizer adapter, outcome-only, no test use in G2."""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--selector',choices=['uct','uniform'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 os.fstat(int(os.environ['NEXT_GPU_LEASE_FD']))
 gate=json.loads((ROOT/'runs/g1/gate.json').read_text());assert gate['pass_gate'] and gate['budget_for_G2']==2
 b3=Path('/home/vipuser/golden-next-20261031/b3/code');assert json.loads((b3/'runs/queue/status.json').read_text())['status']=='B3_seed42_complete'
 cfg=json.loads((ROOT/f'configs/cert_g2_{a.selector}.json').read_text());deadline=json.loads((ROOT/'artifacts/deadline.json').read_text())
 verified=json.loads((ROOT/'artifacts/platform_verification.json').read_text())
 assert verified['rental_expiry_unix']==deadline['rental_expiry_unix'] and time.time()-verified['checked_unix']<3600,'Refresh platform before G2'
 assert time.time()+8*3600<deadline['unix'],'Insufficient predeclared G2 arm window'
 assert shutil.disk_usage(ROOT).free>=16*1024**3
 protocol=json.loads((ROOT/'artifacts/protocol.json').read_text())
 assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in protocol['source_sha256'].items())
 assert json.loads((ROOT/'artifacts/cpu_acceptance.json').read_text())['passed']
 # Independent output root prevents arm collision, including checkpoints and RNG.
 root=ROOT/'runs/g2'/a.selector/'code';root.mkdir(parents=True,exist_ok=True)
 for name in ('train.py','patr_core.py'):
  target=root/name
  if target.exists():assert target.read_bytes()==(ROOT/name).read_bytes()
  else:shutil.copy2(ROOT/name,target)
 for name in ('artifacts','vendor'):
  target=root/name
  if not target.exists():target.symlink_to(ROOT/name,target_is_directory=True)
 config=root/'config.json'
 if config.exists():assert json.loads(config.read_text())==cfg
 else:config.write_text(json.dumps(cfg,indent=2)+'\n')
 import patr_core,train
 from cert_collect import make_collect
 patr_core.ROOT=root;train.ROOT=root
 train.collect=make_collect(patr_core,train.collect)
 original=train.evaluate
 def evaluate(engine,tokenizer,cfg,tasks,dest,seed,split):
  if split=='test':
   result=dict(split='not_evaluated_G2',reason='Test excluded from G2; fixed validation at step 400 and complete curves are reported')
   patr_core.atomic_json(dest,result);return result
  return original(engine,tokenizer,cfg,tasks,dest,seed,split)
 train.evaluate=evaluate
 sys.argv=['train.py','--method','patr','--config','config.json']+(['--resume'] if a.resume else [])
 train.main()
if __name__=='__main__':main()
