"""CPU checks of strict stopping criteria and incomplete/corrupt inputs."""
import copy
from b2_gate import assess

def arm(success, mixed):
    return dict(validation={100:success,125:success,150:success},
                metrics=[dict(step=i,loss=0.,grad_norm=1.,max_parameter_change=1e-6,
                              mixed_group_fraction=mixed,max_leaves=8) for i in range(1,151)])
p,r=arm(70.,.2),arm(69.,.1)
assert assess(p,r)['pass_gate']
q=copy.deepcopy(p);q['validation'][125]=69.
assert not assess(q,r)['pass_gate']
q=arm(70.,.1);assert not assess(q,r)['pass_gate']
q=copy.deepcopy(p)
for row in q['metrics'][:100]:row['mixed_group_fraction']=0
assert assess(q,r)['pass_gate']  # Gate is exactly last50updates.
for kind in ['short','nan','overcap']:
    q=copy.deepcopy(p)
    if kind=='short':q['metrics'].pop()
    elif kind=='nan':q['metrics'][100]['mixed_group_fraction']=float('nan')
    else:q['metrics'][100]['max_leaves']=9
    try:assess(q,r)
    except AssertionError:pass
    else:raise AssertionError(kind)
print('PASS: strict three-checkpoint gate, last50 mixed window, incomplete/NaN/overcap rejection')
