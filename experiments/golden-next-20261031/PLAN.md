# PaTR follow-up execution, 2026-10-31

This supersedes prior deadlines and the old prohibition on new training. User's Oct31 21:45 plan authorizes A/B/C/D/E, with conditional gates and one GPU task at a time. Never resume the completed PaTR500 run.

## Rental and scheduling

Live Galaxy dashboard was refreshed and checked on Oct31: the active 4090PLUS instance expires **2026-11-02 15:09:44 CST**, not Nov3. User specified platform-authoritative time. Do not purchase/renew automatically. Stop starting new compute and checkpoint ongoing training at **Nov2 12:09:44 CST**, reserving the last three hours for verified backups. Central `golden-controls-20261031/code/artifacts/deadline.json` records both timestamps; new workspaces link to it. Do not silently extend beyond a newly verified rental.

Current original trainer PID25916 was preserved exactly (start ticks2771893). Only original CPU parent12947 and CPU history waiter17588 were replaced. Adopter27381 holds the original GPU lock and waits for the same trainer, then execs the original supervisor, which skips finished stages and evaluates GRPO4 supplemental500. History waiter27382 loaded the new deadline and still waits for controls_complete. Handover evidence: controls `runs/control_queue/deadline_handover_20261031.json`. Frozen train/core/config/probe files unchanged.

New workspace: `/home/vipuser/golden-next-20261031/code`; local mirror `work/golden-next-20261031`. Python `/home/vipuser/patr-reproduction/.venv/bin/python`; SSH galaxy-4090. New CPU scheduler PID28302, script `scripts/next_queue.py`, status `runs/queue/status.json`. It waits for BOTH controls_complete and history scheduler status complete, obtains the SAME `golden-controls.../runs/control_queue/supervisor.lock`, confirms GPU empty, then starts work. Children inherit the lock FD to prevent orphan overlap. No current training is paused or restarted. Recovery must first check live PIDs and GPU, not just stale JSON.

Priority under verified shorter rental: **A > B1 > C minimum (PaTR and Random each seed43) > B2/B3 > remaining C > D2 > E1**. Budget estimates are conservative admission checks, not promised completion times. New training requires16GiB free for atomic checkpoint replacement. If space is insufficient, back up and verify new-run artifacts before any justified cleanup; old archives remain immutable. Local disk has >550GiB free; remote initially29GiB free. Never delete evidence to make an unverified backup seem complete.

## Deployed queue (15 stages)

1. A1: four fixed map-generation8/horizon10 evaluations: SFTGRPO400, SFTPaTR400, SFTPaTR500, SFTTreeRandom400. All100 maps CPU-verified identical to the archived map8 protocol;8 differ from original map10. This matches an rLLM parameter ablation, not independently verified official paper data.
2. A2: SFTPaTR400 main100 re-evaluation on4090 with original tasks, seed200042, four samples, decoding0.7/0.8/topk20. Fresh-process/scheduling/numerical effects remain confounded with GPU; report observed rerun delta, not pure hardware causality.
3. A3--A6 summary: reuse PaTR500 supplemental500 (80/81), all-model filtered409, all three pairwise SFTGRPO/PaTR/Random comparisons separately for100/500/409. Task-UID paired bootstrap10,000 reps and map-geometry-cluster sensitivity; intervals exclude training-seed uncertainty. Existing3+1 audit reused.
4. B1 sampling-only: two model processes (original SFT, SFTGRPO400), each24 cells = four K x two selectors x three64-task training batches; then CPU gate.
5. C minimum: original K5, no new leaf cap, independent SFT-initialized seed43 PaTR smoke/400/supp500, then seed43 TreeRandom smoke/400/supp500. New separate workspaces `golden-next-20261031/seed43_{patr,tree_random}/code`. Frozen source byte-identical to controls; config changes ONLY seed42->43. They are baseline replication, not B experiments.

New A/B scripts import frozen controls code read-only. Original train.py, patr_core.py, configs, state_manifest and prior state_probe outputs are not edited. The new routing callable exists only inside B1's process. All model loads read-only; B1 has no optimizer or checkpoint writes. CPU cap tests, exact-map checks, training-only task check, script compilation and queue/config checks passed. A preflight command initially indexed a no-`-u` command incorrectly; fixed in the check, no GPU task was started by that failed preflight.

## B1 prespecified protocol and gate

- Original SFT and SFTGRPO400; K1,2,3,5; initial4, binary expansion, topk2, cumulative maximum8 leaves INCLUDING terminated/pruned leaves. A binary expansion increments this count by1; terminal leaves never refund budget. A nominated branch denied budget continues. Remaining horizon/token cap unchanged.
- Same three training batches from SeedSequence([42,batch+1,1701]),64 unique tasks per batch. Collection RNG20261031+batch, independent of selector/K. No validation/test files are read for choosing K.
- PaTR retains original score pruning; Random disables it; both retain repeated-action pruning. Thus this is NOT a pure selector-only contrast. Count score prunes and ties including/excluding singleton decisions separately.
- Actual trees report forked tasks, mixed success/failure groups, tokens/actions/time and <=8 final leaves. Raw trajectories and routing events are saved.
- Unselected counterfactual outcomes are unobserved in ordinary logs. At the FIRST decision per task containing both selected and unselected parents, copy every parent's real state/history and sample TWO fresh continuations each, without further branching. Keep remaining10-turn horizon, original cumulative token budget and repeat termination. Record shadow cost separately; these samples never alter actual trees or advantages. Shared deterministic shadow seeds exclude method. Estimand: equal-parent mean success for selected minus unselected at that first comparable decision, not full-tree realized descendant success. No assertion that arbitrary shadow states are on-policy training data.
- Gate pools the two prespecified model strata: at least one K must have strictly MORE PaTR mixed groups than Random and a strictly POSITIVE lower95% UID-cluster bootstrap bound for PaTR's shadow selected-minus-unselected success. Repeated UID observations stay in a cluster. Also report each model and Random's shadow delta. Zero or negative intervals fail. Choose highest mixed-group gain among passing K; smaller K breaks ties. Four pointwise intervals are exploratory (no familywise significance claim).
- Both models must finish all48 cells before `runs/b1/gate.json` is valid. Partial results never authorize B2. If failed, skip B2/B3 and continue C. No CPU/fake data are treated as GPU B1 results.

## Conditional later work (not blindly queued)

B2 only after B1 pass and implementation/CPU audit: selectedK/cap8, two independent SFT starts150steps, validation every25. Predeclare “sustained” as PaTR success strictly above Random at ALL final three matched checkpoints100/125/150 AND mean mixed-group proportion over steps101--150 strictly higher. Earlier checkpoints and full cost still reported. This operationalizes the user's qualitative gate before observing B2 data. Tests remain final-only; no test selection.

B3 only after B2 pass: matched400 and later seeds43/44 if sufficient time/space; document whether fresh400 or exact-state150 continuation before starting. No silently changing optimizer, objective, scorer or reward. C remaining baseline seeds43/44 include GRPO8, not substitution byGRPO4. If B1/B2 fails, advance C. Reserve minimum extra PaTR/Random seed first under the shorter platform window.

D1 prior-art review and E3 provenance may run on CPU any time. BranPO already branches by backtracking failed trajectories; do not claim failure-triggered rollback itself is new. D2 three-arm100step experiment must wait for B conclusion and an audited same-leaf-budget protocol. Only training/validation signals gate expansion. E1 eventual no-SFT final-method run is lowest priority. E2 user authorized a routine author inquiry about scorer/topk/Random pruning/code release; verify public recipient and sender account, retain sent receipt, do not disclose unpublished experiment files/results or claim affiliation.

## Current evidence and reporting boundaries

SFTTreeRandom400 is complete: main10079.75/81, supplemental50081/81.6, filtered40977.017/77.751. SFTPaTR500 supplemental500 already80/81; do not rerun A3 unnecessarily. SFTPaTR400 main73.5/75, supplemental75.85/77.6. Single-seed results do not establish superiority across training randomness. Old PaTR fresh-state conditional optimal accuracy is N/A (zero legal denominator), not0% decision accuracy.

3+1 applies631/25600=2.46% task groups; high-minus-low parent mixed-child difference+2.06pp,95%CI[-0.79,4.96]. The often-quoted72% ties includes singleton events; multi-parent ties2038/3192=63.85%. No3+1 training. Filtered409 excludes91 exact labelled geometries in RL training; it does not certify SFT/pretraining disjointness or replace main100/original500.

History probe remains before A:989 eligible/1000,11 skipped, same original manifest;9remainingturns and1024cap versus10/10240 fresh probe, so not pure history causality. Wait for original scheduler completion and verify protected30 hashes and no GPU overlap. Do not alter its protected files.

Every20minutes report Beijing time, phase/count/newsteps, recent train success/invalid means/ranges, latest FIXED validation andstep, finite gradients/actual updates/GPU health, faults and nextstage. No new eval means explicitly say none. Short noisy reward/invalid windows are not computational failure. OOM/NaN/exit/data failures require evidence-preserving repair, never abandon after fail-fast. Back up new raw evidence under `output/golden-next-20261031`, publish lightweight code/config/reports on existing GitHub branch. Weights/optimizer need separate verified local backup, not GitHub.

## Latest closeout preparation (Oct31 22:27 CST)

Protected30 input hashes all match; trainer25916 remains running. GRPO-group4 step46/400, recent20 steps27--46 success mean60.53% (53.52--76.56), invalid1.139% (0--4.247), finite gradients0.381--1.894, positive parameter updates. Fixed validation step40 success66.0 / Pass@4 67.0; training-window statistics are not test results.

E2 author inquiry was SENT at22:24CST and found exactly once in Gmail sent search; do not resend. Local receipt output/golden-next-20261031/author-email-sent.png. PROVENANCE.md records E3 table and D1 BranPO boundary; D1 systematic review remains incomplete.

Local immutable weights/checkpoint backup started PID82945, script code/scripts/backup_completed_local.py (run on Mac), status output/golden-next-20261031/weights-backup/status.json; receipt.json records each SHA-verified file. Five completed SFT-family models plus four complete optimizer checkpoints total approximately33.7GB. It is serial and bandwidth-limited12MB/s, no remote mutation, no GPU. Do NOT claim full backup until status=complete_verified. Active GRPO4 checkpoint is intentionally excluded: add its final model/checkpoint after completion, along with future seed results. Recover transfer failures without deleting source data. No model weights/checkpoints go into GitHub.
