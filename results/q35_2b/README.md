# Qwen3.5-2B experiment records

This is the unified index for the **80GB Qwen3.5-2B** experiments. The repository's new [2B navigation page](https://github.com/EthanY666999/decision-point/blob/main/tracks/2b/README.md) does not move these records. Original FrozenLake results stay on `codex/q35-2b-20261107`; hard-environment results and engineering receipts stay on `codex/q35-hard-env-20261107`, all under `results/q35_2b/`.

## Completed and verified

Rates below are percentages. Fixed validation and final main test are distinct datasets; test is not used for parameter selection. Links open the full records, verification, supplementary evaluations and source evidence on the correct branch.

| Experiment | Environment / method | Training steps | Fixed validation step | Val success / pass@4 | Final main test success / pass@4 |
|---|---|---:|---:|---:|---:|
| [F01_base_eval](https://github.com/EthanY666999/decision-point/tree/codex/q35-2b-20261107/results/q35_2b/F01_base_eval) | FrozenLake original / base | 0 | 0 | 27.25 / 53.00 | 26.50 / 53.00 |
| [F02_noSFT_GRPO4_seed42](https://github.com/EthanY666999/decision-point/tree/codex/q35-2b-20261107/results/q35_2b/F02_noSFT_GRPO4_seed42) | FrozenLake original / noSFT GRPO4 | 400 | 400 | 95.75 / 100.00 | 93.00 / 97.00 |
| [S01_eval_seed42](https://github.com/EthanY666999/decision-point/tree/codex/q35-hard-env-20261107/results/q35_2b/S01_eval_seed42) | Sokoban / base | 0 | 0 | 9.00 / 19.00 | 8.25 / 20.00 |
| [S02_SFT_seed42](https://github.com/EthanY666999/decision-point/tree/codex/q35-hard-env-20261107/results/q35_2b/S02_SFT_seed42) | Sokoban / SFT | 63 | 0 | 37.00 / 56.00 | 43.00 / 61.00 |
| [S03_grpo_seed42](https://github.com/EthanY666999/decision-point/tree/codex/q35-hard-env-20261107/results/q35_2b/S03_grpo_seed42) | Sokoban / SFT GRPO4 | 400 | 400 | 83.00 / 83.00 | 75.25 / 76.00 |

| [S04_tree_random_seed42](https://github.com/EthanY666999/decision-point/tree/codex/q35-hard-env-20261107/results/q35_2b/S04_tree_random_seed42) | Sokoban / SFT RandomK2 | 400 | 400 | 85.00 / 85.00 | 76.50 / 77.00 |

S02's validation step 0 is after 63 SFT training steps. Arithmetic means across all 17 fixed validation points: F02 **86.5294 / 95.1765**; S03 **76.5000 / 78.6471**; S04 **78.0441 / 79.8235**. S04 supplemental 500-task test: **79.30 / 79.80**. Final shortest-path and detour strata are retained with the completed runs.

## Running and remaining work

Snapshot at 2026-11-09 02:55 Beijing: S04 is complete and verified. S05 UCF-v2 has completed step 1/400 and is running step 2. Its fixed validation at step 0 is **37.00 / 56.00**, independently checked against all frozen tasks and four unique lineages. Two startup-only manifest failures were preserved and explicitly excluded from training recovery by the user. The first actual GPU attempt passed validation but failed an allowed-token assertion before any update. A vLLM dependency mask-swap alias defect was reproduced in the actual method on CPU; a two-line clone fix passed 57 swaps and the no-mask case. The one ordinary recovery now passed a full GPU training step: finite gradients, nonzero parameter update, 256 initial plus 128 actual counterfactual leaves, and all 128 forced tokens inside their allowed sets. Original e32ac58 experiment source and configuration remain unchanged. The ordinary recovery allowance is now used. [Repair, authorization and acceptance receipts](stage0/S05_startup_repair_20261109/) preserve the details. This is not a full-run completion or checkpoint fault-recovery acceptance claim. F03 still waits for S05 final acceptance.

The authorized order is S04 → S05 using unchanged `e32ac58`, both with complete final evaluation; then one extra F03 evaluation without SFT retraining; then FLH base/SFT evaluation and GRPO4, RandomK2, UCF seed42; then the remaining queue. CPU loader/handoff receipts are in [stage0/loader_revision_20261108](stage0/loader_revision_20261108/). CPU success does not establish GPU numerical equivalence: the 10-prompt HF/vLLM check, F03-weight FLH smoke checks and GPU fault-recovery end-to-end acceptance remain pending. F37/F38 and S11–S17 are not all in the execution queue.

FLH original-base generated closing-think markers are recorded under the user's provenance-specific exception. Prompt templates must still disable thinking and include empty think blocks. SFT/unknown-origin generated markers remain manual. Both original failed receipts and the later [authorized acceptance](stage0/flh_acceptance/acceptance_authorized_20261108.json) are preserved.

## Storage and publication

This directory contains records, source snapshots, hashes and acceptance receipts, not model weights/checkpoints or large rollout archives. Final models and failure evidence remain on the server. The completed cost checkpoint was SHA-recorded and pruned after independent acceptance; F02's 400 raw rollout files were moved on the same server with SHA-verified links, preserving report compatibility. Small root-owned Conda caches were left intact.

[INDEX.json](INDEX.json) provides machine-readable status and routing. Historical `publication.json` files retain their original commit references. The [supplement manifest](publication_batches/20261108_supplement_manifest.json) identifies the added records and their SHA-256 hashes. No result commit is merged into a running worktree.
