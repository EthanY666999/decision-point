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

Snapshot at 2026-11-09 06:17 Beijing: S04 is complete and verified. S05 UCF-v2 has completed step 82/400 and is running step 83. Its latest fixed validation at step 75 is **74.75 / 77.00**, independently checked against all frozen tasks and four unique lineages. Two startup-only manifest failures were preserved and explicitly excluded from training recovery by the user. The first actual GPU attempt passed validation but failed an allowed-token assertion before any update. A vLLM dependency mask-swap alias defect was reproduced in the actual method on CPU; a two-line clone fix passed 57 swaps and the no-mask case. The one ordinary recovery now passed a full GPU training step: finite gradients, nonzero parameter update, 256 initial plus 128 actual counterfactual leaves, and all 128 forced tokens inside their allowed sets. Original e32ac58 experiment source and configuration remain unchanged. The ordinary recovery allowance is now used. [Repair, authorization and acceptance receipts](stage0/S05_startup_repair_20261109/) preserve the details. This is not a full-run completion or checkpoint fault-recovery acceptance claim. F03 still waits for S05 final acceptance.

The latest authorized order is S05 on unchanged `e32ac58` through final verification → F03 single extra evaluation → **S13 noSFT-GRPO4 Sokoban seed42** → **UCF-v2.1 offline/CPU/one-step GPU admission** → **S15 noSFT-UCF-v2.1 Sokoban seed42** → H01/H02 hard FLH evaluations → H03/H04/H05 → original suffix. S14 stays at its original later P2 position. S13/S15 use original Qwen3.5-2B, reply cap4096, 400 steps and validation every25 steps. Only S15 and F10 enable `--fork_point first_fence`; S05 and all SFT-UCF retain v2. **Initialization and implementation both differ between noSFT UCF-v2.1 and SFT UCF-v2; this is not a strict remove-SFT-only ablation.**

**UCF-v2.1 admission is BLOCKED; no GPU smoke or S15/F10 has started.** F02's complete 400-step training audit finds 96.51% fence coverage, 96.50% candidates with successor budget, and calibrated expected forks2.0/task (mathematical aggregate, not GPU measurement). First-fence action versus the unchanged final parser mismatches **41328/322741 = 12.8053%**, exceeding the fixed **2%** gate. The prefix/fence simulation passes on four-action support (z1.484), but fails with residual p[0]=0.08 (z149.442; agreement with analytically biased expectation z1.931). The specified residual/fallback rules remain unchanged. A third literal prefix-causality audit also fails: unchanged v2 whole-batch calibration lets a later candidate on the same trajectory change the current rho from0.5 to0.4 despite an identical current prefix and w. Fixed-c_step unit tests do not prove full-pipeline prefix invariance; the unchanged controller is preserved and this failure is a mandatory gate. **101 functional CPU tests and byte-identical S05 selection replay through81 steps do not constitute overall v2.1 CPU or GPU acceptance.** F03/S13 remain authorized; S13 will be audited after completion, and the serial queue then stops at S15 admission until the definition is revised. [Implementation, scope and derivation](../../docs/q35_ucf_v21_20261109.md), [failed acceptance and evidence](stage0/ucf_v21_20261109/acceptance.json).

S11/S12/S14/S16/S17 and F37/F38 remain outstanding. F03 loading and actual-F03 FLH smoke checks remain pending.

FLH original-base generated closing-think markers are recorded under the user's provenance-specific exception. Prompt templates must still disable thinking and include empty think blocks. SFT/unknown-origin generated markers remain manual. Both original failed receipts and the later [authorized acceptance](stage0/flh_acceptance/acceptance_authorized_20261108.json) are preserved.

## Storage and publication

This directory contains records, source snapshots, hashes and acceptance receipts, not model weights/checkpoints or large rollout archives. Final models and failure evidence remain on the server. The completed cost checkpoint was SHA-recorded and pruned after independent acceptance; F02's 400 raw rollout files were moved on the same server with SHA-verified links, preserving report compatibility. Small root-owned Conda caches were left intact.

[INDEX.json](INDEX.json) provides machine-readable status and routing. Historical `publication.json` files retain their original commit references. The [supplement manifest](publication_batches/20261108_supplement_manifest.json) identifies the added records and their SHA-256 hashes. No result commit is merged into a running worktree.

S13/S15 fixed-validation reporting compares the same step: success/pass@4, mixed/all-failure task counts among100 tasks with4 lineages, and mean generated reply tokens per trajectory. Missing peer steps remain pending, and training leaf statistics are separate. Runtime loader commit: `95b78572c702322369dc0cb0d5cd3f061d569b3e`; v2.1 admission is separately blocked.
