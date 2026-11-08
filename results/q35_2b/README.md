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

Snapshot at 2026-11-09 07:17 Beijing: S05 historical UCF-v2 completed107/400 and was running108, unchanged PID295755/controller295732/source e32ac58. Recent5 steps135.41 seconds/step, GPU100%,70.08GiB. Latest fixed validation is **step100:79.75 / 83.00**, mixed6/all-failure17/all-success77, mean reply27.09 tokens; the 100 frozen tasks and four unique lineages were independently recomputed. Final test has not started. The one ordinary recovery is already used. [Earlier repair/authorization receipts](stage0/S05_startup_repair_20261109/) remain historical; no further automatic rerun is authorized.

Authorized serial order: **S05 full verification → F03 single extra evaluation → H01/H02 hard FLH evaluations → H03 SFT-GRPO4 → H04 SFT-Random K=2 → H05 corrected SFT-UCF → S13 new-protocol noSFT-GRPO4 → v2.1 one-step GPU admission → S15 new-protocol noSFT-UCF**, followed by the unchanged suffix. S14 stays later P2. S13/S15 keep original Qwen3.5-2B, seed42, cap4096,400 steps and fixed validation every25.

The user revised the protocol and approved future SFT-UCF corrections. Future noSFT training/evaluation ends after the **first sampled token completing a complete action block**, with indivisible boundary-token overhang recorded and later tokens discarded; SFT response cutoff stays disabled. Nodes remain immediately after the first fence, without skipping whitespace tokens. Corrected UCF excludes the entire original consequence class from the full vocabulary, including residual tokens where applicable, and uses leave-one-question calibration. **The historical0.5→0.4 dependence came from the same trajectory; S05 also has this defect and remains unmodified.**

**116 CPU tests pass; GPU acceptance is still pending.** The corrected p[0]=0.08 simulation has max z**1.59655**, analytical bias0; the same-question calibration audit passes. Default-off SFT IDs/masks/logprobs/actions match the historical path, and S05 selection through100 replays byte-for-byte. Original failures are preserved. S13's actual new-protocol400-step training audit must still meet the unchanged**2%** threshold before v2.1 GPU/S15 admission. F02 remains old-protocol historical evidence and is no longer the paired noSFT control. Initialization, protocol and/or implementation differ across the relevant SFT/noSFT comparisons; this is not a strict remove-SFT-only ablation. See [current definition](../../docs/q35_ucf_v21_revision_20261109.md), [current CPU and pending-GPU acceptance](ucf_v21_revision_20261109/acceptance.json), and [original failed evidence](stage0/ucf_v21_20261109/acceptance.json).

S11/S12/S14/S16/S17 and F37/F38 remain outstanding. F03 loading and actual-F03 FLH smoke checks remain pending.

FLH original-base generated closing-think markers are recorded under the user's provenance-specific exception. Prompt templates must still disable thinking and include empty think blocks. SFT/unknown-origin generated markers remain manual. Both original failed receipts and the later [authorized acceptance](stage0/flh_acceptance/acceptance_authorized_20261108.json) are preserved.

## Storage and publication

This directory contains records, source snapshots, hashes and acceptance receipts, not model weights/checkpoints or large rollout archives. Final models and failure evidence remain on the server. The completed cost checkpoint was SHA-recorded and pruned after independent acceptance; F02's 400 raw rollout files were moved on the same server with SHA-verified links, preserving report compatibility. Small root-owned Conda caches were left intact.

[INDEX.json](INDEX.json) provides machine-readable status and routing. Historical `publication.json` files retain their original commit references. The [supplement manifest](publication_batches/20261108_supplement_manifest.json) identifies the added records and their SHA-256 hashes. No result commit is merged into a running worktree.

S13/S15 fixed-validation reporting compares the same step: success/pass@4, mixed/all-failure task counts among100 tasks with4 lineages, and mean generated reply tokens per trajectory. Missing peer steps remain pending, and training leaf statistics are separate. Runtime loader commit: `e633a28c8599327de55801569123b34f40ad1d48`; corrected CPU checks pass, while new-protocol S13 audit and GPU admission remain pending.
