"""CPU-only checks of queue ordering, readiness and technical admission."""
import json
from pathlib import Path
import tempfile
import b3_queue as q

assert q.stages() == [('patr', True), ('patr', False), ('tree_random', True), ('tree_random', False)]
with tempfile.TemporaryDirectory() as tmp:
    q.ROOT = Path(tmp) / 'b3'
    q.B2 = Path(tmp) / 'b2'
    q.save(q.B2 / 'runs/queue/status.json', {'status':'B2_complete'})
    q.save(q.B2 / 'runs/analysis/gate.json', {'pass_gate':True})
    q.NEXT = Path(tmp) / 'next'
    q.CONTROLS = Path(tmp) / 'controls'
    q.save(q.NEXT / 'runs/queue/status.json', {'status':'running'})
    q.save(q.CONTROLS / 'runs/control_queue/status.json', {'status':'controls_complete'})
    q.save(q.CONTROLS / 'runs/audit/state_probe_hist1/scheduler_status.json', {'status':'complete'})
    assert not q.prerequisites()
    q.save(q.NEXT / 'runs/queue/status.json', {'status':'A_B1_Cminimum_complete'})
    assert q.prerequisites()
    q.save(q.B2 / 'runs/analysis/gate.json', {'pass_gate':False})
    assert not q.prerequisites()
    q.save(q.B2 / 'runs/analysis/gate.json', {'pass_gate':True})
    assert not q.complete('patr', True)
    p = q.ROOT / 'runs/smoke/patr_seed42'
    q.save(p / 'status.json', {'status':'smoke_complete','step':1,'total_steps':1})
    assert not q.complete('patr', True)  # Completion requires saved final evaluation.
    q.save(p / 'final_test.json', {})
    cfg = {'model':'fixture-sft'}
    q.save(q.ROOT / 'configs/b3_k2_400.json', cfg)
    q.save(p / 'b3_identity.json', {'smoke':True,'method':'patr','config':cfg,'source_sha256':{}})
    row = dict(step=1,loss=0.,grad_norm=1.,max_parameter_change=1e-6,max_leaves=8,leaf_cap=8,task_groups=4)
    (p / 'train_metrics.jsonl').write_text(json.dumps(row)+'\n')
    q.audit_smoke('patr')
    assert q.read(p / 'technical_acceptance.json')['passed']
    for key, value in [('grad_norm', float('nan')), ('max_parameter_change', 0), ('max_leaves',9)]:
        bad = dict(row); bad[key] = value
        (p / 'train_metrics.jsonl').write_text(json.dumps(bad)+'\n')
        try:
            q.audit_smoke('patr')
        except AssertionError:
            pass
        else:
            raise AssertionError(f'Unsafe smoke accepted: {key}')
print('PASS: C-first readiness, independent arm order, completed artifact checks and NaN/no-update/overcap rejection')
