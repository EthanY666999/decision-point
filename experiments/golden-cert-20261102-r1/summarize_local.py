import json,math,sys
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parents[2];out=root/'output/golden-cert-20261102-r1';selector=sys.argv[1] if len(sys.argv)>1 else 'uct';previous=int(sys.argv[2]) if len(sys.argv)>2 else 0
p=out/f'runs/g2/{selector}/code/runs/formal/patr_seed42';rows=[json.loads(x) for x in (p/'train_metrics.jsonl').read_text().splitlines()] if (p/'train_metrics.jsonl').exists() else [];recent=rows[-20:];delta=[r for r in rows if r['step']>previous]
s=dict(selector=selector,step=rows[-1]['step'] if rows else 0,previous_step=previous,new_completed_steps=len(delta),recent_n=len(recent))
s['recent']={k:dict(mean=sum(r[k] for r in recent)/len(recent),min=min(r[k] for r in recent),max=max(r[k] for r in recent)) for k in ['success_rate','invalid_action_fraction','mixed_group_fraction'] if recent}
s['increment_cost']={k:sum(r.get(k,0) for r in delta) for k in ['input_tokens','generated_tokens','executed_actions','invalid_actions','rollout_seconds','step_seconds']}
s['probe_increment']={k:sum(r.get('probe_cost',{}).get(k,0) for r in delta) for k in ['input_tokens','generated_tokens','executed_actions','seconds']}
s['recent_distributions']={}
for k in ['p_hat_distribution','fork_turns']:
 c=Counter()
 for r in recent:c.update(r.get(k,{}))
 s['recent_distributions'][k]=dict(c)
s['dup_first_action_rate']=[r.get('dup_first_action_rate') for r in recent]
s['gradient_finite']=all(math.isfinite(r['grad_norm']) for r in recent) if recent else None
s['parameter_changes']=[r.get('max_parameter_change') for r in recent]
v=sorted(p.glob('validation_*.json'));s['validation']=[dict(file=f.name,**json.loads(f.read_text())) for f in v[-2:]]
print(json.dumps(s,indent=2));(out/f'latest_{selector}_monitor.json').write_text(json.dumps(s,indent=2)+'\n')
