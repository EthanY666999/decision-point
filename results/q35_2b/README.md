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
| [S05_ucf_seed42](https://github.com/EthanY666999/decision-point/tree/codex/q35-hard-env-20261107/results/q35_2b/S05_ucf_seed42) | Sokoban / SFT UCF-v2 (historical) | 400 | 400 | 85.50 / 87.00 | 81.50 / 84.00 |

S02's validation step 0 is after 63 SFT training steps. Arithmetic means across all 17 fixed validation points: F02 **86.5294 / 95.1765**; S03 **76.5000 / 78.6471**; S04 **78.0441 / 79.8235**. S04 supplemental 500-task test: **79.30 / 79.80**. S05's 17-point mean is **77.9706 / 81.1765** and supplemental 500-task test is **84.80 / 86.00**. Final shortest-path and detour strata are retained with the completed runs.

## Resumed after S05

S05 completed all 400 steps and final main100/supplemental500 evaluations; verification and independent raw-record checks passed. The user instructed: “跑完S零五整个队列节奏暂停掉吧。” At 2026-11-09 18:44 Beijing, both actual handoff and pause status report **paused_by_user_after_S05**, GPU processes are absent, GPU/supervisor locks are released, and all 45 deferred fences are retained. The user subsequently explicitly requested continuing the queued work at about 19:00 Beijing. That new instruction supersedes the pause: **the original serial queue has resumed**, beginning with F03's single additional evaluation. All original acceptance gates remain mandatory. The original failed attempts and one used ordinary recovery remain preserved.

See [S05 results](S05_ucf_seed42/README.md), [final completion audit](S05_ucf_seed42/analysis/S05_final_completion_verified.json), [pause result](pause_after_S05_20261109/queue_pause_status.json), and [pause implementation](../../docs/q35_pause_after_S05_20261109.md). The final model remains on the same server, and its active checkpoint was pruned only after successful final verification.

Latest user priority: **F03 single extra evaluation (completed) → H01/H02 hard FLH evaluations (completed) → S06 Sokoban GiGPO → H03 SFT-GRPO4 → H04 SFT-Random K=2 → H05 SFT-UCF-v2.2 → S13 new-protocol noSFT-GRPO4 → v2.2-ff one-step GPU admission → S15 new-protocol noSFT-UCF**, then the unchanged suffix. The 2026-11-09 resume instruction restores execution authorization without bypassing predecessors. S14 stays later P2. S13/S15 retain original Qwen3.5-2B, seed42, cap4096, 400 steps and fixed validation every25.

The latest implementation is **UCF-v2.2**, with **UCF-v2.2-ff** for future noSFT. It uses question-held-out calibration while preserving leave-one-trajectory p-hat/baselines. The user's exact test (alter only trajectory j's suffix, final reward and later candidates; keep all other trajectories fixed) passes: c_k and unchanged-prefix rho on j stay fixed, while other three rho may change.

STAY complements use allowed_token_ids. Non-STAY complements use **unrestricted n=256, max_tokens1, fixed-seed proposals**, taking the first token outside the original consequence class; there is no logit_bias. Exhaustions produce the requested fallback and are counted. SFT also uses opening-fence conditional probabilities, explicitly authorized as a **third difference from historical S05**, alongside calibration and sampling support. NoSFT-ff additionally permits reasoning before the first fence and uses the first-complete-action-block token cutoff in training and evaluation. Boundary tokens, including inseparable extra characters, are retained and trained; all subsequent tokens are discarded.

**122 CPU tests pass; actual GPU acceptance remains pending.** The256-proposal simulation has max z**2.28696** at p[0]=0.08, with zero observed exhaustions in the tested cases. Finite256 exhaustion plus original-only fallback is not a universal exact-unbiasedness guarantee; its probability and possible bias are documented. S13 must still pass the unchanged**2%** actual new-protocol full400-step training audit before the v2.2-ff GPU gate and S15.

F02's full raw replay finds the first fence is not the first complete action-block start in **18297/319697=5.72323%** of complete-block turns. It is historical reference, not a new-protocol paired control. S05's stored selection weights through116 give7424 ratios c_k/c_batch: median**0.9971085**, P5–P95**0.9845580–1.0232206**, range**0.9797766–1.1279932**. S05 is unmodified.

Production software is **torch2.10.0 / vLLM0.19.1 / transformers5.22.2**. Metadata was backfilled to38 existing2B run directories with original-byte backups and provenance; past loaded versions are not inferred from current installations. Future2B starts record versions directly. See [UCF-v2.2 definition](../../docs/q35_ucf_v22_20261109.md), [CPU and pending-GPU acceptance](ucf_v22_20261109/acceptance.json), [c ratio distribution](ucf_v22_20261109/S05_c_ratio_summary.json), and [software backfill](ucf_v22_20261109/hardware_versions_backfill.json). All prior v2.1 receipts remain historical.

S11/S12/S14/S16/S17 and F37/F38 remain outstanding. F03 loading and actual-F03 FLH smoke checks remain pending.

FLH original-base generated closing-think markers are recorded under the user's provenance-specific exception. Prompt templates must still disable thinking and include empty think blocks. SFT/unknown-origin generated markers remain manual. Both original failed receipts and the later [authorized acceptance](stage0/flh_acceptance/acceptance_authorized_20261108.json) are preserved.

## Storage and publication

This directory contains records, source snapshots, hashes and acceptance receipts, not model weights/checkpoints or large rollout archives. Final models and failure evidence remain on the server. The completed cost checkpoint was SHA-recorded and pruned after independent acceptance; F02's 400 raw rollout files were moved on the same server with SHA-verified links, preserving report compatibility. Small root-owned Conda caches were left intact.

[INDEX.json](INDEX.json) provides machine-readable status and routing. Historical `publication.json` files retain their original commit references. The [supplement manifest](publication_batches/20261108_supplement_manifest.json) identifies the added records and their SHA-256 hashes. No result commit is merged into a running worktree.

S13/S15 fixed-validation reporting compares the same step: success/pass@4, mixed/all-failure task counts among100 tasks with4 lineages, and mean generated reply tokens per trajectory. Missing peer steps remain pending, and training leaf statistics are separate. Current isolated runtime commit: `6e8f0591639afdc674f428edca1baa0d96367ddc` (S06 queue priority; all 58 experiment configurations unchanged). The earlier UCF-v2.2 122-test CPU receipt retains its scope; new-protocol S13 audit and GPU admission remain pending and await their serial positions. S05 completed on unchanged `e32ac58b8a857eb98023ec1f79bbe0cf855c95ac`.

Resume evidence: [authorization](resume_after_S05_20261109_190132/authorization.json), [CPU/control preflight](resume_after_S05_20261109_190132/cpu_acceptance.json), [composite CPU receipt](resume_after_S05_20261109_190132/cpu_functional_composite.json). The original 122-test algorithm receipt is retained byte-for-byte; algorithm/protocol source is unchanged, and six pause-control tests passed on the current tree. This is not a claim that 128 tests ran together or that future GPU gates have passed. The prior pause and S05-completion records remain historical evidence.

## S06 GiGPO priority, 2026-11-09

The user explicitly selected “Sokoban：S06 GiGPO（接着S05做基线）”. S06 uses the same S02 SFT checkpoint as S03/S04/S05, seed42, 400 steps, gamma0.95, omega1, and fixed validation every25. The isolated runtime changes queue order only; 85 CPU identity/order/precondition checks passed. Previous algorithm/control tests are inherited only for byte-identical files, not claimed as newly rerun GPU tests.

The old scheduler had already passed the H03 entry check before the priority fence was installed. Its unintended H03 startup was cancelled with zero completed training steps and no checkpoint; the complete directory and original journal were preserved as user cancellation, not an experiment failure or recovery. F03/H01/H02 completed verification and are not rerun. See [authorization](priority_S06_GiGPO_20261109/authorization.json), [CPU checks](priority_S06_GiGPO_20261109/cpu_acceptance.json), and [cancellation evidence](priority_S06_GiGPO_20261109/H03_user_priority_cancellation.json).
