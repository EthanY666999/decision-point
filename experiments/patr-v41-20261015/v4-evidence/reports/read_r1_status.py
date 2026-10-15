import json
from pathlib import Path
root=Path('/home/vipuser/patr-reproduction-v4');run=root/'runs/r1-20261015-a';snap=root/'snapshots/r1-20261015-a'
result={}
for key,p in [('status',run/'status.json'),('progress',run/'progress.json'),('r0_gate',snap/'reports/r0_gate.json'),('identity',snap/'frozen_identity.json')]:
 result[key]=json.loads(p.read_text()) if p.exists() else None
result['metrics']=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()] if (run/'metrics.jsonl').exists() else []
result['checkpoint_tests']=(root/'reports/checkpoint_integrity_round5.log').read_text()
result['new_cpu_tests']=(root/'reports/contracts_round6.log').read_text()
print(json.dumps(result))
