"""Descriptive fixed-endpoint summary; no 150-step stopping or test selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import patr_core as core
out={}
for selector in ('uct','uniform'):
    r=ROOT/f'runs/g2/{selector}/code/runs/formal/patr_seed42'
    s=json.loads((r/'status.json').read_text());assert s['status']=='complete' and s['step']==s['total_steps']==400
    rows=[json.loads(x) for x in (r/'train_metrics.jsonl').read_text().splitlines()];assert [x['step'] for x in rows]==list(range(1,401))
    vals={str(i):json.loads((r/f'validation_{i:04d}.json').read_text())['success_rate_pct'] for i in range(0,401,25)}
    out[selector]=dict(validation_success_pct=vals,last3_mean=sum(vals[str(i)] for i in (350,375,400))/3,
      mixed101_150=sum(x['mixed_group_fraction'] for x in rows[100:150])/50,
      mixed351_400=sum(x['mixed_group_fraction'] for x in rows[350:400])/50)
core.atomic_json(ROOT/'runs/g2/summary400.json',dict(arms=out,val400_difference_pp=out['uct']['validation_success_pct']['400']-out['uniform']['validation_success_pct']['400'],no_test_use=True,no_150_gate=True,no_multiseed_claim=True))
