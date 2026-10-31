# B2 protocol fixed after the training-only B1 gate

Recorded 2026-11-01 03:32 CST. Status: protocol fixed; isolated sampling adapter and CPU behavioral checks implemented; workspace/config and CPU gate evaluator prepared; serial scheduling and GPU smoke audit pending. This is not a GPU launch instruction. The existing seed43 PaTR and Tree-Random replication queue retains priority and must not be interrupted.

B1 completed all 48 cells and selected K=2 using the previously fixed rule. At K=2, PaTR had 118 mixed groups versus Random112 across384 model-task observations (+6, +1.5625percentage points; paired UID-bootstrap95%CI [0,3.3854]pp). PaTR selected-minus-unselected shadow continuation success was +5.0080pp,95%CI[1.7965,8.3333]. This is an exploratory admission gate, not confirmation of a method advantage. SFT contributed113vs107 mixed groups; trained SFTGRPO contributed5vs5. K1andK5also passed, K3didnot; K2had the greatest mixed-group gain. No validation or test results were used for K selection.

## Two matched short training runs

- Independent starts from the exact original Golden Trace SFT checkpoint, actor/reference/vLLM initialized together. Never initialize from GRPO, PaTR400/500 or smoke final weights.
- K=2, cumulative leaf cap8, initial4, binary expansion, top-k2. Ended and pruned leaves retain their consumed budget. If a branch cannot expand because the cap is reached, it continues normally. Use the same audited cap semantics as B1.
- PaTR and Tree-Random each150updates, seed42, fixed validation every25updates. All other settings, prompts, reward, loss, optimizer, training data and evaluation decoding inherit the existing frozen controls configuration. Explicit budget/evaluation cadence changes are part of this protocol.
- Preserve PaTR score pruning, Random score-pruning disabled, repeated-action pruning retained in both. Consequently this is not a pure selection-only ablation. Count score prunes separately.
- New workspaces, new configs, record source hashes and config diffs. Frozen controls and seed43 runs remain unchanged. The training adapter may install the cap router in its own process; no shadow sampling enters the RL trajectory groups or policy loss.
- Record per-update mixed success/failure groups, forked-task fraction, ties, actual generated tokens, input tokens, executed actions, timing, invalid actions, finite gradients and nonzero parameter updates. Sampling costs are not guaranteed equal merely because leaf ceilings match.

## Admission, completion and next gate

CPU tests must cover terminal/pruned leaves consuming budget, multiple expansions under one task cap, continuing unexpanded branches, and no shadow contamination. Technical smoke must verify finite gradients, real parameter updates, cap compliance and independent SFT initialization. A noisy one-batch reward is not grounds for cancellation.

Only launch after the current C minimum queue has completed, with the shared GPU lease and no live GPU owner. Check the central deadline and reserve enough time and disk for both arms and backups. Insufficient capacity means record pending; do not reduce the150update budget or interrupt an existing run silently.

Predeclared B2 pass: PaTR fixed-validation Success is strictly higher than Random at all three matched checkpoints100,125,150, AND PaTR's mean mixed-group fraction over updates101–150 is strictly higher. Earlier validation, full costs, and uncertainty remain reported. No test-based selection or optional stopping. If either condition fails, stop B2/B3 expansion and report the failure.

B3 is not launched by this document. Only after B2 passes, predeclare fresh400 training versus exact-state continuation and available seed budget before implementation. The existing rental boundary is unchanged: stop compute by2026-11-02 12:09:44CST.


## CPU implementation audit, 2026-11-01 03:54 CST

New b2/code/run_b2.py entrypoint installs b2_collect.make_collect into a separately copied trainer. The adapter creates a fresh BudgetRouter per training batch, capture=False, leaves evaluation untouched, restores the routing callable even after an exception, checks returned group cardinalities against cumulative leaf counts, and adds mixed/fork/prune metrics without changing policy updates. run_b2 requires a GPU lease descriptor, prepared B1-derived protocol/config, and stable code identity across resume. It is not currently runnable as a complete experiment: frozen runtime/config/data and supervisor/gate still need preparation/audit. No GPU smoke or B2 training has run.

CPU test_b2_collect.py passed on remote Python without importing torch/vLLM: fresh-batch budgets for both selectors, cap-blocked continuation, terminal leaves consuming budget, shadow capture disabled, evaluation bypass, exception restoration. Python compilation passed. Source train.py/patr_core.py and running seed43 training remain unchanged. Full runtime smoke must still verify sampled probabilities, actual gradient update and cap accounting before admission.


## Gate evaluator audit, 2026-11-01 04:13 CST

b2_gate.py reads only training metrics and fixed validation100/125/150. It refuses incomplete150-step arms, NaNs, leaf counts above8, and mismatched configs/source/protocol. Equal validation or equal mixed-group means fails the strict gate. CPU tests passed for positive case, equality failures, exact101--150window and incomplete/NaN/overcap rejection. Synthetic test fixtures are not experimental evidence. Full workspace and serial supervisor plus realGPU technicalsmoke remain pending; no B2 job was launched.


## Workspace preparation, 2026-11-01 04:32 CST

Remote /home/vipuser/golden-next-20261031/b2/code now has byte-identical frozen train.py/patr_core.py copies, vendor/data/build-deps links to the original resources, and the central deadline link. New configs/b2_k2_150.json differs from the original only in branch_interval5->2, new leaf_cap8, steps400->150, eval_every40->25; original seed42 and SFT model path remain. Both arms share this config and independent method output directories. B1 gate and dataset hashes are recorded in artifacts/b2_protocol.json and preparation.json. No run directory exists and no model was loaded.

Both CPU test suites passed in this isolated directory. An additional audit of all48 B1 cells/3072 task groups confirmed returned leaf counts equal initial4 plus cumulative binary expansions, including terminal/pruned leaves. This supports the adapter's cardinality assertion on actual frozen-collector outputs. Serial admission/lease supervisor and GPU technicalsmokes remain pending; existing C queue remains first.


## Serialized scheduling audit, 2026-11-01 05:12 CST

The isolated b2_queue.py is deployed as CPU waiter49561. It waits for A_B1_Cminimum_complete and the completed controls/history statuses, then takes the same exclusive supervisor.lock and requires an empty GPU. It verifies B1 gate/data hashes, admits each stage only with16GiB free and enough time for all remaining arms (3.5h per150-step arm,15min per smoke,15min margin), and honors the central deadline. No automatic deletion or budget reduction; capacity shortages require evidence-preserving recovery. Smoke acceptance checks finite gradients, positive updates, cap8, config/source identity before its formal arm. Both arms independently initialize the original SFT; existing checkpoints resume unchanged. The final gate cannot start B3 automatically.

Remote CPU adapter, gate and scheduler tests passed; current training44087 remains the sole GPU owner. Waiter status is waiting_for_Cminimum and it has not initialized CUDA. Actual GPU smoke audit remains pending until the C pair finishes.
