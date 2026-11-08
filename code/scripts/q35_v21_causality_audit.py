"""CPU causal audit: legacy same-question leakage and corrected held-out c."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'ucf')]
from ucf_select import calibrate
from ucf_corrected import leave_question_out_constants

def audit():
    original=[.5]*4
    literal=[calibrate([w]) for w in [original, original+[.5]]]
    # 64 questions, four trajectories per question; fixed current prefix/w.
    a={ti:[.5]*4 for ti in range(64)}
    b={ti:list(ws) for ti,ws in a.items()};b[0]+=[.5]
    c={ti:list(ws) for ti,ws in a.items()};c[1]+=[.5]
    old=[calibrate(list(x.values())) for x in [a,b]]
    corrected=[leave_question_out_constants(x)[0] for x in [a,b,c]]
    same=corrected[0]==corrected[1]
    return dict(passed=same,requirement='current prefix/w and other trajectories feature values fixed; changing own-question candidate suffix does not change c or rho',
        original_literal=dict(c=literal,rho=[x*.5 for x in literal],changed_source='later candidate on the SAME trajectory/question, not another question'),
        original_64_questions=dict(c=old,rho=[x*.5 for x in old],same_question_dependence=old[0]!=old[1]),
        corrected=dict(c=corrected,rho=[x*.5 for x in corrected],same_question_invariant=same,other_question_change_allowed=corrected[0]!=corrected[2]),
        S05_legacy_affected=True,S05_modified=False,GPU_used=False,
        feature_scope='w/ph/b remain original leave-one-trajectory feature rule; other three trajectory outcomes may change w. c excludes all nodes of own question.',
        source_sha256={f:hashlib.sha256((ROOT/'ucf'/f).read_bytes()).hexdigest() for f in ['ucf_select.py','ucf_corrected.py']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
