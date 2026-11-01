"""Read-only remote health and independently audited local lightweight snapshots."""
import collections, datetime, gzip, json, math, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
previous, stamp = int(sys.argv[1]), sys.argv[2]
base = ROOT / 'output/golden-next-20261031/b3/runs/formal/patr_seed42'
out = ROOT / 'output/golden-cert-20261101' / ('monitor_' + stamp)
out.mkdir(exist_ok=True)
rows = [json.loads(l) for l in (base/'train_metrics.jsonl').read_text().splitlines()]
n = rows[-1]['step']; window = rows[-20:]; cycle = [r for r in rows if r['step'] > previous]
h = {'last_step': n, 'previous_step': previous, 'new_steps': n-previous,
     'window': [window[0]['step'], n], 'metrics': {},
     'cycle': {k: sum(r[k] for r in cycle) for k in ['generated_tokens','input_tokens','executed_actions','episodes','step_seconds','rollout_seconds']},
     'all_steps_valid': all(math.isfinite(r['grad_norm']) and math.isfinite(r['max_parameter_change']) and r['max_parameter_change'] > 0 and r['max_leaves'] <= 8 for r in rows),
     'steps_contiguous': [r['step'] for r in rows] == list(range(1,n+1))}
for k in ['success_rate','invalid_action_fraction','mixed_group_fraction','grad_norm','max_parameter_change','step_seconds']:
    values = [r[k] for r in window]
    h['metrics'][k] = dict(mean=statistics.mean(values), min=min(values), max=max(values))
count = 0
for r in window:
    groups = collections.defaultdict(list)
    for line in gzip.open(base/f"rollouts_{r['step']:04d}.jsonl.gz", 'rt'):
        x = json.loads(line); count += 1
        assert len(x['ids']) == len(x['mask']) == len(x['old_logprobs'])
        assert all(math.isfinite(v) for v in x['old_logprobs'])
        groups[x['task']['uid']].append(x['reward'])
    assert len(groups) == 64
    assert sum(min(v) < max(v) for v in groups.values()) == r['mixed_groups']
h['raw_window_trajectories_verified'] = count
vp = max(base.glob('validation_*.json'))
h['validation_step'] = int(vp.stem.split('_')[-1]); h['validation'] = json.loads(vp.read_text())
vr = [json.loads(l) for l in vp.with_suffix('.jsonl').read_text().splitlines()]
groups = collections.defaultdict(list)
for r in vr: groups[r['task']['uid']].append(r['reward'])
success = sum(r['reward'] > 0 for r in vr); passed = sum(max(v)>0 for v in groups.values())
assert len(groups)==100 and len(vr)==400 and all(len(v)==4 for v in groups.values())
assert 100*success/len(vr)==h['validation']['success_rate_pct']
assert 100*passed/len(groups)==h['validation']['pass_at_4_pct']
h['validation_raw'] = dict(successful_trajectories=success, passed_tasks=passed,
    actions=sum(len(r['trace']) for r in vr), invalid_actions=sum(t['action'] not in (1,2,3,4) for r in vr for t in r['trace']))
protected = json.loads((ROOT/'output/golden-controls-20261031/state_probe_hist1/protected_inputs_sha256.json').read_text())
script = '''import json,pathlib,hashlib,subprocess,shutil,time,datetime
P=pathlib.Path('/home/vipuser/golden-controls-20261031/code')
expected=EXPECTED
out={'time':time.time(),'cst':datetime.datetime.now().isoformat(),'protected30':{k:hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in expected.items()},'disks':{k:shutil.disk_usage(k)._asdict() for k in ['/','/data']}}
for k,c in {'gpu':['nvidia-smi','--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu','--format=csv,noheader'],'owners':['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],'processes':['ps','-p','146367,151967','-o','pid,ppid,lstart,args']}.items():out[k]=subprocess.check_output(c,text=True)
for k,p in {'cert_queue':'/home/vipuser/golden-cert-20261101/code/runs/queue/status.json','b3_queue':'/home/vipuser/golden-next-20261031/b3/code/runs/queue/status.json','handover':'/home/vipuser/golden-cert-20261101/code/runs/queue/handover.json','deadline':str(P/'artifacts/deadline.json')}.items():out[k]=json.loads(pathlib.Path(p).read_text())
proc=pathlib.Path('/proc/146367/stat');out['trainer_start_ticks']=proc.read_text().split()[21] if proc.exists() else None
out['arm_status']=json.loads(pathlib.Path('/home/vipuser/golden-next-20261031/b3/code/runs/formal/patr_seed42/status.json').read_text())
cp=pathlib.Path('/home/vipuser/golden-next-20261031/b3/code/runs/formal/patr_seed42/checkpoint.pt');out['checkpoint_metadata']={'bytes':cp.stat().st_size,'mtime':cp.stat().st_mtime}
print(json.dumps(out))
'''.replace('EXPECTED',repr(protected))
h.update(json.loads(subprocess.check_output(['ssh','galaxy-4090','/home/vipuser/patr-reproduction/.venv/bin/python -'], input=script, text=True)))
h['backups'] = {p.parent.name:json.loads(p.read_text()).get('status') for p in (ROOT/'output/golden-next-20261031').glob('weights-backup*/status.json')}
(out/'health.json').write_text(json.dumps(h,indent=2)+'\n')
print(json.dumps({k:v for k,v in h.items() if k not in ['protected30','handover','b3_queue','backups']},indent=2))
assert all(h['protected30'].values()) and h['all_steps_valid'] and h['steps_contiguous']
