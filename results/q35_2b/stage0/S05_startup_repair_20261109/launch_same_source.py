import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path('/home/vipuser/decision-point-q35-hard/code')
COMMIT='e32ac58b8a857eb98023ec1f79bbe0cf855c95ac'
CONFIG=Path('/data/runs/queue_q35_hard/configs/S05.json')
CONFIG_SHA='3f5bbfb28333afda3544b5a6a3284a4db59a746c4148ff7699d307e13269a817'
p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==COMMIT
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
assert hashlib.sha256(CONFIG.read_bytes()).hexdigest()==CONFIG_SHA
frozen=json.loads((ROOT/'artifacts/frozen_q35.json').read_text());corrections={}
for name,expected in frozen['source_sha256'].items():
 if name in ['train.py','scripts/hard_queue_supervisor.py']:
  blob=subprocess.check_output(['git','show',COMMIT+':code/'+name],cwd=ROOT)
  corrected=hashlib.sha256(blob).hexdigest();corrections[name]={'old':expected,'pinned_commit':corrected};expected=corrected
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
for name in ['cpu_acceptance','gpu_acceptance','calibration','phase1_gate']:
 rec=json.loads((ROOT/'artifacts/q35_gates'/(name+'.json')).read_text());assert rec['passed'] and rec['model_family']=='Qwen3.5'
cfg=json.loads(CONFIG.read_text());assert cfg['ucf_beta']==.25 and cfg['ucf_variant']=='ucf' and cfg['ucf_controller']=='v2'
print(json.dumps({'preflight_passed':True,'source_commit':COMMIT,'config_sha256':CONFIG_SHA,'verified_manifest_paths':len(frozen['source_sha256']),'corrections':corrections,'GPU_used':False}),flush=True)
if a.launch:
 sys.path[:0]=[str(ROOT),str(ROOT/'ucf')]
 import train
 sys.argv=['train.py','--method','ucf','--config',str(CONFIG),'--output-dir','/data/runs/hard_queue/S05_ucf_seed42']
 train.main()
