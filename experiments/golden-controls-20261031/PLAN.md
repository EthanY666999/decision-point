# Golden Trace control experiment execution record

Updated 2026-10-31. User explicitly requested the Claude-proposed audits and new controls, then requested GitHub synchronization. This authorizes NEW controls after the completed PaTR500 experiment. Never resume the old500 run.

## Execution

- Host alias galaxy-4090; remote workspace /home/vipuser/golden-controls-20261031/code.
- Python /home/vipuser/patr-reproduction/.venv/bin/python.
- Queue scripts/controls_supervisor.py; check runs/control_queue/status.json, logs, live PID and GPU before recovery. Exclusive flock prevents duplicate supervisors.
- Serialized order: six-model fixed-state greedy diagnostics; TreeRandom smoke and400; fixed supplemental500; GRPO group4 smoke and400; supplemental500.
- Both new RL arms start independently from the same original SFT weights, fresh optimizer, frozen SFT reference. Formal configs preserve original400 settings except method or flat_group_size4. train.py has only mmap=True CPU-load engineering change.
- Existing TreeRandom also disables score pruning; original400 has4 score-prune events. Report this composite control explicitly.
- GRPO4 is a group-size control, NOT exact token/compute matching. Report actual token/action/time budgets. Old400 hardwareA100; new4090 walltime not directly comparable.
- Hard rental safety deadline Nov1 04:55 CST. Do not remove/extend. Safe SIGTERM at training boundary saves checkpoint. Restore only same config with --resume, no duplicate GPU owner. Paused/failed queue requires diagnosis and evidence-preserving recovery, not blind replay.
- Probe output dirs must be absent for a fresh run. On incomplete inference preserve directory under a timestamped failed-attempt name before rerunning that model. Same frozen state manifest, decoding and model required.
- Smoke covers one4-task batch; no learning signal there alone is not sufficient reason to abandon formal training. Inspect finite values and actual changes over a longer window before conclusions.
- Monitor every20minutes and report Chinese status even if healthy: time, stage/steps, new steps, recent train success/invalid ranges, fixed validation metrics/step, gradient/update/GPU health, fault action and next stage. Training rates are not fixed test rates.

## Audited findings

CPU offline_audit.py completed. Original400 has4183 fork events over25600 task groups (16.34%);3029 same-score events includes991 singleton events. True multi-parent ties2038/3192=63.85%, not72%. Two selected parents with different scores631/25600=2.46%; extra100 is125/6400=1.95%. Paired high-minus-low mixed-child outcome difference2.06pp, UID-cluster95%CI[-0.79,4.96]; extra100 -0.8pp CI[-6.45,4.84]. Higher scores associate with higher success, not demonstrated higher contrast. No3+1 training queued.

Exact labelled-map equality confirms91 supplemental tasks overlap all10000 RL training tasks. Remaining409: oldPaTR67.30/72.86 success/Pass4; SFT18.40/40.59; SFTGRPO40064.43/65.04; SFTPaTR40070.48/72.62; SFTPaTR50075.73/76.77. Sensitivity analysis, never replace original500. No claim all409 absent from SFT/pretraining. Zero-invalid conditional metric foroldPaTR undefined because no eligible trajectories.

Fixed-state probe:1000 unique(task,state) pairs sampled round-robin from originaltest100; safe reachable states withgoal distance1..10; all models receive exact same fresh standard context with10 remainingsteps, nohistory. ManifestSHA256 cd0cdc2d39c164a5c79eef6a1a635a59f156dc73daf1c791146daef68b944e08. Greedy temperature0, original10240 outputtoken cap; report long/truncated outputs rather than silently shorten cap. Conditional optimal-action accuracy requires valid denominator and coverage; compute common-valid subset if nonempty. Test diagnostics never choose training config.

## Next phase gate

Finish controls and analyze actual sampling costs, mixed reward groups and fixed validation before spending on new methods. BranPO already recursively backtracks incorrect trajectories; ReTRAIL/SPROUT/LEAFE also make broad novelty claims untenable. Candidate contribution must be narrowed to a tested sampling/point-selection mechanism, not failure-backtracking itself. A100step three-arm pilot requires a frozen leaf-cap6 protocol, same initial4, common task draws, correct restored history/environment/remaining budget, true old logprob and explicit shared-prefix treatment. Distinguish equalmaximumleaf budget from equalactualtokens. Use validation/training signal to decide400; reserve tests for final comparisons. Additionalseeds conditional on useful pilot, time and rental bounds. No unreviewed rewind training is queued.

Probe completion: run code/scripts/summarize_probes.py after completed model evaluations. Its state_probe_report.json applies production token-limit rejection and common-legal comparisons; raw model summary.json is syntactic parsing only. Do not report truncated outputs as executed legal actions.
