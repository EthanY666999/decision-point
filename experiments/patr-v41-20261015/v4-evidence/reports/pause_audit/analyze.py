import json,gzip,sys,collections,time
from pathlib import Path
r=Path('/home/vipuser/patr-reproduction-v4');s=r/'snapshots/smoke-20261015-a';sys.path.insert(0,str(s))
from training_contract import filter_and_advantages
o=r/'runs/patr-400-20261015-a';attempt=next(o.glob('attempt-*'));reports=[]
for step in [60,68,69,70,71,72,73,76,80,85]:
 p=attempt/f'raw_{step-1}.jsonl.gz'
 if not p.exists():continue
 with gzip.open(p,'rt') as f:bs=[json.loads(l) for l in f]
 groups=[[b for b in bs if b['task_id']==i] for i in range(64)]
 rows,rejected=filter_and_advantages([[b['reward'] for b in g] for g in groups]);z=collections.Counter()
 for g,j,a in rows:
  b=groups[g][j];positive=a>0;tag='positive' if positive else 'negative';ids=b['ids'];mask=b['mask'];assert len(ids)==len(mask)
  n=sum(mask);eos=sum(bool(m) and t==151645 for t,m in zip(ids,mask));z[tag+'_rows']+=1;z[tag+'_tokens']+=n;z[tag+'_eos']+=eos;z[tag+'_abs_adv_token_mass']+=abs(a)*n;z[tag+'_abs_adv_eos_mass']+=abs(a)*eos
  for t in b['trace']:z[tag+'_finish_'+str(t.get('finish_reason'))]+=1
 z['retained_groups']=64-len(rejected);z['step']=step;reports.append(dict(z))
p=r/'reports/pause_audit';p.mkdir(exist_ok=True);(p/'token_advantage_audit.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports))
