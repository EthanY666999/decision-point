"""Exact predeclared G2 validation/training gate; no test files read."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import patr_core as core
B2=Path('/home/vipuser/golden-next-20261031/b2/code/runs/formal/patr_seed42')

def decision(uct,uniform,b2,uct_mixed,uniform_mixed,b2_mixed):
 return dict(three_validation_strict_wins=all(uct[s]>uniform[s] for s in (100,125,150)),
             three_validation_not_below_B2=all(uct[s]>=b2[s] for s in (100,125,150)),
             mixed_strictly_above_both=uct_mixed>uniform_mixed and uct_mixed>b2_mixed)
def main():
 vals={};mixed={}
 for name,run in {'uct':ROOT/'runs/g2/uct/code/runs/formal/patr_seed42','uniform':ROOT/'runs/g2/uniform/code/runs/formal/patr_seed42','b2':B2}.items():
  status=json.loads((run/'status.json').read_text());assert status['status']=='complete' and status['step']==150
  rows=[json.loads(l) for l in (run/'train_metrics.jsonl').read_text().splitlines()]
  assert [r['step'] for r in rows]==list(range(1,151))
  vals[name]={s:json.loads((run/f'validation_{s:04d}.json').read_text())['success_rate_pct'] for s in (100,125,150)}
  mixed[name]=sum(r['mixed_group_fraction'] for r in rows if 101<=r['step']<=150)/50
 checks=decision(vals['uct'],vals['uniform'],vals['b2'],mixed['uct'],mixed['uniform'],mixed['b2'])
 core.atomic_json(ROOT/'runs/g2/gate.json',dict(pass_gate=all(checks.values()),checks=checks,validation=vals,mixed101_150=mixed,no_test_use=True))
if __name__=='__main__':main()
