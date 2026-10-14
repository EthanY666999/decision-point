Prepared candidates only. The frozen training files have NOT been changed.

base_pending.patch:
- schema4 runtime provenance: exactly the approved 41 selected execution files,
  8 distribution versions, actual RECORD and METADATA SHA. Includes XFORMERS and
  tokenizers and the actual custom vLLM source; never trusts RECORD alone.
  All selected files, including the three native extensions, are fully read on
  every fingerprint call. No runtime-source SHA cache or shared-venv write.
  Install prefixes/timestamps are excluded from ID. Read-races and missing
  required installed files abort. Scope is not the entire dependency/CUDA stack.
- Old schema3 resume is explicitly rejected before metadata writes. Original
  fingerprints, checkpoints and source snapshots are never upgraded in place.
- DAPO candidate means include mean_training_reward and first_turn_invalid_fraction.
- Before sync, require idle requests/queues/block tables/allocated blocks. After
  sync, require reset_prefix_cache() is exactly True. Log success or raise with
  state before any successful return; no silent reset failure.
- evaluate requires idle state before sampling and again before writing results.

conditional_eval_reset.patch:
- Adds reset at evaluate entry. This is NOT automatically approved by CPU tests;
  root must decide after the full fresh/warm/fresh GPU diagnostic. The complete
  alternative source is train_with_eval_reset.py.txt.

Prepared real-file CPU tests:
- reports/test_pending_runtime_execution.py: same-version/metadata/length and
  restored-mtime byte mutation in vLLM, XFORMERS and tokenizers; refusal before
  run artifact writes; distribution bytes/missing file/read-race; portable prefix;
  original schema3 artifacts preserved. All package locations are real temporary
  PathDistribution files outside fixture source roots. Shared venv never edited.
- reports/test_pending_inference_boundary.py: busy engine refused before weight
  load; reset failure abort; idle sync return interface; failed evaluation leaves
  original output intact and no pending-request result is written.
- reports/test_pending_dapo_aggregate.py: real main, 2/3 candidate batches with
  rejected and accepted groups, costs and weighted means verified.

Evidence (in-memory proposals only, production source hashes unchanged):
delivery_candidates_cpu_no_eval_flush_v2.txt: 20 passed, 1 conditional skip, 0.60s.
delivery_candidates_cpu_conditional_flush_full.txt: original 98 CPU tests + 21
new checks = 119 passed in 60.30s. No GPU work performed.
First delivery_candidates_cpu_no_eval_flush.txt has one fixture error: appending
an empty CSV row made importlib.metadata reject RECORD. Corrected to a valid
duplicate source row; the original failed log is retained. This was not a
production training failure.

Release procedure (only after explicit root release):
1. Confirm frozen original SHA values still match manifest.json; keep the original
   short_study_frozen_source.zip and every historical run fingerprint untouched.
2. Apply base and only the approved conditional patch; move/adapt tests into tests/.
   The boundary test uses train._delivery_candidate_evaluation_reset solely as an
   in-memory runner flag. Replace those assertions/conditional skip with the
   selected final behavior; do not introduce this diagnostic flag into production.
3. Run actual complete CPU suite against files on disk, not only the in-memory
   runner. New delivery source freeze is separate from original study source.
4. Do not claim runtime provenance changed retroactively, sampling repeatability
   solved without GPU evidence, or all 26 scientific/protocol issues resolved.
