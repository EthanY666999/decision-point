# S06 Sokoban GiGPO final result

Completed 400 steps from the original S02 SFT checkpoint (seed 42, gamma 0.95, omega 1). Training and final verification passed. Test sets were evaluated only after training; they were not used for parameter selection.

| Evaluation | Tasks x samples | Success | Pass@4 |
|---|---:|---:|---:|
| Fixed validation at 400 | 100 x 4 | 84.50% | 86.00% |
| Main test | 100 x 4 | 86.50% | 87.00% |
| Supplemental test | 500 x 4 | 82.35% | 82.80% |

Arithmetic mean of the 17 fixed validation points (0, 25, ..., 400): success 76.3235294118%, pass@4 78.5882352941%.

Independent completion audit checked all 17 validation raw files and both final test sets against frozen tasks and unique four lineages, recomputed exact shortest-path and detour strata, independently ran BFS on 700 evaluation maps, checked all 400 finite nonzero updates, 62 verification-record hashes, and the retained final model SHA. The last five training steps averaged 113.90398 seconds.

The final model remains on the authorized 80G host at `/home/vipuser/q35-final-models/S06_gigpo_seed42`, with the original logical symlink retained. The checkpoint was pruned only after successful verification. No weights are published here. `verification.json` is preserved byte-for-byte, including its pre-publication `github_publication: pending` field; this Git commit and the separate on-host publication receipt record the later publication.

Files: [independent audit](completion_verified.json), [verification](verification.json), [full validation mean and final strata](additional_metrics.json), [run report](run_report.json), [model retention](model_storage.json), [checkpoint pruning](checkpoint_pruning.json).
