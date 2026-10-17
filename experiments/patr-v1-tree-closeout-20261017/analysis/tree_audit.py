import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
root=Path('/home/vipuser/patr-v1-tree-20261016');out=root/'reports/closeout-20261017'
result={}
for method in ['patr','tree_random']:
 rows=[json.loads(x) for x in (root/'runs/formal'/f'{method}_seed42/branch_events.jsonl').read_text().splitlines()]
 assert [r['step'] for r in rows]==list(range(1,401))
 windows={}
 for label,lo,hi in [('all',1,400),('first80',1,80),('last80',321,400)]:
  kinds=Counter();forks=[];diff=[];live=0
  for row in rows:
   if not lo<=row['step']<=hi:continue
   ee=row['events'];kinds.update(e['kind'] for e in ee)
   ff=[e for e in ee if e['kind']=='fork_candidates'];forks.extend(ff)
   live+=len({e['task_id'] for e in ff})
   leaves=[e for e in ee if e['kind']=='leaf']
   for f in ff:
    selected=[];unselected=[]
    for c in f['candidates']:
     desc=[e for e in leaves if e['task_id']==f['task_id'] and (e['lineage']==c['lineage'] or e['lineage'].startswith(c['lineage']+'.'))]
     assert desc
     val=np.mean([e['reward']>0 for e in desc]);(selected if c['selected'] else unselected).append(val)
    if selected and unselected:diff.append(float(np.mean(selected)-np.mean(unselected)))
  comparable=[f for f in forks if any(c['selected'] for c in f['candidates']) and any(not c['selected'] for c in f['candidates'])]
  def tied(f):return max(c['score'] for c in f['candidates'])-min(c['score'] for c in f['candidates'])<1e-12
  windows[label]=dict(iterations=hi-lo+1,sampled_tasks=64*(hi-lo+1),tasks_with_live_fork=live,all_forks=len(forks),comparable_forks=len(comparable),all_tied_forks=sum(tied(f) for f in forks),all_tied_comparable_forks=sum(tied(f) for f in comparable),selected_minus_unselected_descendant_success_mean=float(np.mean(diff)) if diff else None,event_counts=dict(kinds))
 result[method]=windows
(out/'tree_training_audit.json').write_text(json.dumps(dict(limitation='Observational; score-based pruning, descendant counts, and changing policies confound branch outcome differences. This is not a controlled causal score-quality test.',results=result),indent=2)+'\n')
print(json.dumps(result))
