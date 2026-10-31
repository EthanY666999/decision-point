# B2 protocol fixed after the training-only B1 gate

Recorded 2026-11-01 03:32 CST. Status: protocol fixed; isolated sampling adapter and CPU behavioral checks implemented; complete workspace preparation, gate evaluator, serial scheduling and GPU smoke audit pending. This is not a GPU launch instruction. The existing seed43 PaTR and Tree-Random replication queue retains priority and must not be interrupted.

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
