Unique latest candidate set; prepared ONLY, not applied to production source.
Supersedes prior delivery_candidates* directories for future application.

base_pending.patch contains runtime schema4 selected actual-file identities,
two DAPO episode-weighted logging means, mandatory idle/reset-failure guards,
and official verified-tokenizer/text checks with exact incremental fallback.

Choose at most one conditional evaluation policy after controlled GPU evidence
and explicit root release:
- conditional_eval_reset.patch: reset before evaluation only.
- conditional_eval_isolation.patch: reset before AND after evaluation. Apply to
  the base candidate, not on top of conditional_eval_reset.patch.
The corresponding complete train candidates are also included. Entry+exit
prevents evaluation-cache execution state from reaching subsequent training
collect calls; it is not a fix for demonstrated validation-label leakage.

Latest CPU validation:131 passed in58.38s. This was an in-memory candidate run
with entry+exit reset enabled, not a source edit. Frozen train.py,
run_integrity.py and scripts/verify_final.py hashes all remained unchanged.
Full log:reports/delivery_candidates_eval_isolation_cpu_full.txt.
Conditional absence may skip policy-specific tests; an actual files-on-disk
full suite is still required after the final approved policy is applied.

pending_tests/ contains exact tested snapshots and a real UTF-8-tail fixture.
Tests currently contain diagnostic-only policy flags set by the in-memory
runner. On application, replace those flag branches with assertions of the
chosen final policy; do not add diagnostic flags to production training code.
The verifier test accesses the preserved fixture in reports/fixtures/.

Old study schema3 fingerprints, checkpoints, raw evidence and original source
ZIP stay unchanged. Delivery receives its own schema4 source freeze and a
single production-main integration smoke. See DELIVERY_SMOKE_ACCEPTANCE_PLAN.txt.
