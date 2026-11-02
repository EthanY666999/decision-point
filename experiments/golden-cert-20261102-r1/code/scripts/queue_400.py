"""Authorized r1: reference acceptance with new seeds, then independent 400-step arms."""
import fcntl,json,time
from queue_v2 import ROOT,Q,CONTROL,admission,run,validate,status,complete,gpu

def main():
    with (Q/'scheduler.lock').open('a+') as single:
        fcntl.flock(single,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate()
        gate=json.loads((ROOT/'runs/g1/gate.json').read_text())
        assert gate['pass_gate'] and gate['reference_acceptance_not_required_for_g1']
        admission('CERT_r1_reference_acceptance_and_G2_400_pair',17.25)
        with (CONTROL/'runs/control_queue/supervisor.lock').open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert not gpu()
            result=ROOT/'runs/acceptance/sft_patr400/result.json'
            if not result.exists():
                if result.parent.exists():result.parent.rename(result.parent.with_name('sft_patr400_incomplete_'+str(int(time.time()))))
                run('acceptance_sft_patr400_new_seeds',['scripts/acceptance.py','--model','sft_patr400'],lease,3600)
            if not json.loads(result.read_text())['passed']:
                status('technical_acceptance_failed',model='sft_patr400',no_training=True);return
            for selector in ('uct','uniform'):
                dest=ROOT/f'runs/g2/{selector}/code/runs/formal/patr_seed42'
                if complete(dest/'status.json'):
                    assert json.loads((dest/'status.json').read_text())['step']==400
                    continue
                # The initial platform evidence may be older than an hour after recovery.
                if selector=='uniform' or time.time()-json.loads((ROOT/'artifacts/platform_verification.json').read_text())['checked_unix']>=3500:
                    admission('CERT_r1_G2_400_'+selector,8.25 if selector=='uniform' else 16.25)
                cmd=['run_cert.py','--selector',selector]
                if (dest/'checkpoint.pt').exists():cmd+=['--resume']
                run('g2_400_'+selector,cmd,lease,8*3600)
            run('g2_400_summary',['scripts/summary_400.py'],lease,300)
        status('G2_400_pair_complete',no_150_step_gate=True,summary=str(ROOT/'runs/g2/summary400.json'))
if __name__=='__main__':
    try:main()
    except BlockingIOError:raise SystemExit('Already locked; no restart')
    except Exception as e:
        status('error_evidence_retained',error=repr(e));raise
