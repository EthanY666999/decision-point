"""One bounded storage-only recovery; same source/config/checkpoint identity."""
import runpy,json,time,sys
from pathlib import Path
ROOT=Path('/home/vipuser/patr-reproduction-v43')
sys.path.insert(0,str(ROOT/'reports'))
import run_pipeline as p
out=ROOT/'runs/patr-400-v43-a'
try:
 cursor=json.loads((ROOT/'reports/monitor_cursor.json').read_text())
 p.cursors=cursor['cursors'];p.streaks=cursor['streaks']
 if p.stage('patr',out,400,resume=True):
  assert (out/'evaluations/reference_test_400.json').exists()
  p.write(p.REPORT,dict(state='completed_patr',updated=time.time(),grpo='pending_time_and_storage_review'))
except Exception as exc:
 p.write(p.REPORT,dict(state='controller_error_requires_review',error=repr(exc),updated=time.time()))
 raise
