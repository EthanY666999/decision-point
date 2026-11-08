"""Literal rho-prefix-invariance counterexample using the unchanged controller."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ucf'))
from first_fence import locate
from ucf_select import calibrate

def audit():
    # Same current node + three other-trajectory nodes. Only the suffix of the
    # current trajectory changes: a later turn now contains another first fence.
    fence=999;history=[1,90,fence,30,fence]
    later_without=[50,51,52];later_with=[50,fence,40,fence]
    current_end=locate(history,0,len(history),fence)
    assert current_end==3
    weights=[.5]*4
    results=[]
    for suffix in [later_without,later_with]:
        future_has_node=locate(suffix,0,len(suffix),fence) is not None
        ws=weights+([.5] if future_has_node else [])
        c=calibrate([ws],target=2.)
        results.append(dict(current_prefix=history[:current_end],current_w=.5,
                            future_candidates=int(future_has_node),c_step=c,
                            current_rho=min(1.,max(.03,c*.5))))
    return dict(passed=results[0]['current_rho']==results[1]['current_rho'],
      requirement='arbitrary tokens after the current node must not change current rho',
      cases=results,reason='v2 whole-batch calibration reads later candidate support from the same trajectory; fixed-c_step unit invariance is insufficient',
      controller_changed=False,GPU_used=False,source_sha256=hashlib.sha256((ROOT/'ucf/ucf_select.py').read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit();a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
