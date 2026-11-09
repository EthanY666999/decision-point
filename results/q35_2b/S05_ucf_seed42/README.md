# S05: Sokoban SFT UCF-v2, seed 42

Completed 400 training steps and all final evaluations; verification passed. The user paused the entire subsequent queue after S05. F03 and all later experiments remain unstarted under this pause; new explicit authorization is required to resume.

| Evaluation | Tasks x lineages | Success (%) | Pass@4 (%) |
|---|---:|---:|---:|
| Fixed validation, step 400 | 100 x 4 | 85.50 | 87.00 |
| Arithmetic mean of all 17 fixed validation points (0,25,...,400) | 17 points | 77.970588 | 81.176471 |
| Final main test | 100 x 4 | 81.50 | 84.00 |
| Final supplemental test | 500 x 4 | 84.80 | 86.00 |

Fixed validation, training samples and final tests are distinct. No test result selected parameters. The final-completion audit independently checked all 17 raw validation sets, all final task/lineage records, frozen maps and shortest-path/detour metadata for 700 evaluation tasks, 1,262 recorded hashes, the model SHA and all 400 finite/nonzero parameter updates. See [final audit](analysis/S05_final_completion_verified.json), [strata](additional_metrics.json) and paired task bootstrap comparisons against [S03](analysis/S05_vs_S03_paired_final.json) and [S04](analysis/S05_vs_S04_paired_final.json).

Training step 400 had 256 initial leaves in 64 groups: 52 all-success / 4 all-failure / 8 mixed. All 380 actual leaves gave 41 / 4 / 19. Pseudo training rows and fallback placeholders are excluded. Its CF parse mismatch was zero. Earlier nonzero CF parse observations remain in historical evidence and are not erased by this final-step value.

S05 remained on original runtime e32ac58b8a857eb98023ec1f79bbe0cf855c95ac and historical UCF-v2. It is not UCF-v2.2: future v2.2 changes calibration to question-held-out c, full-complement sampling, and opening-fence conditional probabilities. Known historical same-question suffix dependence is retained, not retroactively corrected. The single ordinary recovery and original failed-attempt evidence remain recorded.

The final model is retained on the same server at /home/vipuser/q35-final-models/S05_ucf_seed42 through the original path's symlink; model SHA256 is fed786bc4f0b498b29d27e9b1f0852438a10cbd4202d6c376f884955bf55dbdc. The 22,582,327,849-byte active checkpoint was removed only after successful final verification. No weights or checkpoints were uploaded.

Raw training and evaluation trajectories remain on the server. final_outcomes.csv is an exact compact projection of all 2,400 final trajectories; training_metrics_summary.csv projects all 400 completed metric rows. Original record hashes are preserved in verification.json; its github_publication=pending records the original verification time, and publication is documented by a separate later receipt.

GPU and supervisor locks were released, GPU processes are absent, and queue_pause_status.json plus handoff_status.json both report paused_by_user_after_S05. All 45 deferred handoff fences remain intact. The legacy controller's completed_and_handed_off means the guarded entry point was called; no later experiment launched.
