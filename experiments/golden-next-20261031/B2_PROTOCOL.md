# B2 protocol fixed after the training-only B1 gate

Recorded 2026-11-01 03:32 CST. Status: protocol fixed; implementation and audit pending. This is not a GPU launch instruction. The existing seed43 PaTR and Tree-Random replication queue retains priority and must not be interrupted.

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
