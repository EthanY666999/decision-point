import json,shutil,sys,time,subprocess,hashlib
from pathlib import Path
root=Path('/home/vipuser/patr-reproduction-v4')
new=root/'snapshots/smoke-20261015-a'
assert not new.exists(), 'Snapshot already exists; inspect before retry'
for name in ['qwen_update_acceptance.json','update_integration_acceptance.json']:
 r=json.loads((root/'reports'/name).read_text());assert r['passed'],name
shutil.copytree(root/'snapshots/r1-20261015-a',new,ignore=shutil.ignore_patterns('__pycache__'))
for p in root.glob('*.py'):shutil.copy2(p,new/p.name)
shutil.copytree(root/'tests',new/'tests',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['qwen_update_acceptance.json','update_integration_acceptance.json','contracts_round7.log']:
 shutil.copy2(root/'reports'/name,new/'reports'/name)
sys.path.insert(0,str(new))
from r1_diagnostics import fingerprint
cfg=json.loads((new/'config_candidate.json').read_text());identity=fingerprint(cfg)
(new/'frozen_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
gate=json.loads((new/'reports/r0_gate.json').read_text())
gate.update(stage='R0_for_independent_GRPO_smoke',fingerprint_sha256=identity['sha256'],created_unix=time.time())
for name in ['qwen_update_acceptance.json','update_integration_acceptance.json','contracts_round7.log']:
 gate['evidence_sha256'][name]=hashlib.sha256((new/'reports'/name).read_bytes()).hexdigest()
gate['remaining_limits'][-1]='Independent GRPO smoke and formal training not yet accepted'
(new/'reports/r0_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({'snapshot':str(new),'fingerprint':identity['sha256'],'stage':'frozen_not_launched'}))
