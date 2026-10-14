Latest combined candidate, prepared only; no frozen source has been changed.

base_pending.patch includes the prior runtime schema4/DAPO/inference-idle guards
from ../delivery_candidates/base_pending.patch, plus scripts/verify_final.py:
the official CLI verifies original tokenizer/model manifest SHA and every
nonweight tokenizer/config asset against original manifest/fingerprint before
loading actual AutoTokenizer with local_files_only=True. It decodes every saved
response ID list with skip_special_tokens=True and cleanup=False, rejecting any
text mismatch, and reports asset identity plus explicit decode verification.
The reusable audit_evaluation helper retains an optional tokenizer for lightweight
fixtures/legacy callers; absent decoder is explicitly reported as unverified.
The official CLI always supplies the verified loader for intermediate and final
evaluations. Decoder library versions are recorded; asset proof must not be
misdescribed as an exhaustive loaded-library byte audit.

conditional_eval_reset.patch is STILL conditional on the GPU cache diagnostic
and root's decision. No evaluation/weight sampling behavior was changed in files.
The original study freeze and schema3 run artifacts remain intact.

Six additional regressions in reports/test_pending_verifier_token_text.py cover
actual CLI paths, truthful optional-helper scope, original asset loading,
same-size tokenizer bytes/manifest mutation refusal, and same-length response IDs
that remain consistent with masks/trace but decode Left while saved text says
Right. The latter passes the old count/parser-only check and is rejected by the
candidate decoder before writing outputs.

reports/delivery_candidates_with_token_audit_cpu_full.txt:
125 passed in62.13s for original98+new27 CPU checks with proposed evaluation reset
enabled IN MEMORY only. Actual run_integrity.py/train.py/scripts/verify_final.py
SHA remained unchanged. No GPU execution. Previous119 candidate and reports are
retained. After root authorizes release, apply the chosen candidate, adapt/move
the four pending test files into tests (remove the diagnostic-only reset flag as
explained in ../delivery_candidates/README.txt), then run the actual on-disk full
CPU suite and the independent delivery-main GPU smoke.
