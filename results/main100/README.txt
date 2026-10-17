V1 revised-tree PaTR and Tree-Random, seed42 — final evidence

Both methods completed 400 training iterations and 800 optimizer updates. No further training is authorized or queued by this closeout. The training monitor was deleted on 2026-10-17.

Start with analysis/REPORT.txt (Chinese), analysis/results.csv, and analysis/training_summary.png.
Main test Success / Pass@4 (%): V1 GRPO 61 / 62; new PaTR 64.5 / 71; new Tree-Random 64.75 / 68.
PaTR minus Random Success: -0.25 percentage points; paired 100-task bootstrap 95% CI [-8.75, 8.25]. One training seed; no demonstrated superiority or equivalence.
Additional map-generation max_steps=8 test (interaction horizon still 10): GRPO 65.25 / 67; PaTR 68.75 / 79; Random 70.75 / 73. Eight map geometries changed. Re-evaluation also changes some sampled traces on unchanged maps, so protocol deltas are not a pure eight-map intervention estimate.
All three main-test models produce invalid first actions on all 400 trajectories; the original environment permits continued interaction after invalid output. Scores were not repaired or redefined.

Contents:
- source/: frozen training/tree implementation, task data, config, runtime override and vendored environment/agent.
- runs/: complete train metrics, all validation and final test trajectories, compressed tree events, status and source snapshots for both new methods.
- v1_baselines/: original final tests and provenance for GRPO, PaTR and Random.
- reports/: execution events, completion manifests, GRPO-model diagnostic and earlier stage evidence.
- analysis/: independent closeout scripts, map audit, bootstrap, integrity validation and supplemental raw test outputs.
- raw_rollout_manifest.json: SHA256 for all 800 remote raw training rollout files (not bundled here).
- MANIFEST.json: file hashes of the remote lightweight snapshot. Local publication additions, including this index and the archive receipt, are outside that snapshot manifest.
- REMOTE_ARCHIVE_RECEIPT.json: verified remote/local tar SHA256.

No model weights or optimizer state are committed to Git. Model/checkpoint hashes and remote provenance are in analysis/integrity.json. Any separately completed local model backup is documented by a subsequent explicit receipt, not implied by this lightweight archive.
