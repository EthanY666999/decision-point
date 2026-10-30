# Golden Trace warmup experiment — 2026-10-31

User-approved order: Golden Trace SFT, GRPO baseline acceptance, then PaTR. Both RL jobs start independently from the same SFT checkpoint; PaTR is not initialized from the GRPO final model. Actor, frozen reference and vLLM use the same SFT checkpoint, with a fresh RL optimizer. Archived train.py and patr_core.py are unchanged.

## Current execution

- SSH alias: galaxy-a100 (A100 40 GB).
- Remote repository: /home/vipuser/golden-sft-patr-20261031
- Python: /home/vipuser/patr-reproduction/.venv/bin/python
- Source base: GitHub EthanY666999/decision-point commit 806985abb6cca3d7ae0c14adce285ec3c90f7bb7.
- Local branch: codex/golden-sft-patr-20261031 (not yet published).
- Status: code/runs/golden_sft/pipeline_status.json and stage-specific logs.
- Supervisor PID at launch: 3379; verify live command before signaling. Old supervisor 2876 was terminated to reorder the queue; its SFT child 3044 was preserved and completed.
- SFT complete: 1,000 trajectories, 19,676 action/end-of-turn targets, longest sequence 2,376 tokens; 1 epoch, seed42, batch16/microbatch1, LR 2e-5, 63 updates, 109.07 seconds, peak allocated memory 9.245 GiB. Full FP32 parameters, BF16 autocast, causal SDPA, no truncation or packing. Token-normalized batch loss includes action text and EOS only.
- Model/data SHA checks, all 1,000 environment replays, all six unit tests, full-dataset mask audit passed. GRPO smoke completed with actual parameter updates and model export.
- Queue: GRPO smoke -> SFT standalone test -> GRPO400 -> diagnostics -> technical acceptance -> PaTR smoke -> PaTR400 -> diagnostics. Frozen RL config unchanged except model path and experiment label. Two separate formal output directories, fresh optimizers. No old deadline file copied.

## Overnight operation and recovery

User explicitly authorizes autonomous debugging, recovery and further evidence-based PaTR experiments while asleep, until superseded by a new instruction. Do not ask approval for routine fixes. An ACTIVE thread heartbeat named Golden Trace：GRPO 验收后运行 PaTR checks every ten minutes; remote queue runs independently of local SSH.

Do not stop because of a few bad batches, fluctuating reward, zero-variance groups, or invalid actions. Judge sustained trends, finite gradients, parameter updates and raw trajectories. The old successful runs also had first-turn invalid actions; strict parsing and original scoring must remain intact. Do not repair responses with added backticks or rescore them to inflate results.

For genuine OOM, NaN, process exits or data corruption: preserve logs/checkpoints, diagnose, repair and resume. A supervisor failure is a request for investigation, not permission to abandon the task. If changing only documented runtime memory/batch options, preserve scientific config. Never silently change reward/scorer/hyperparameters. The trainer checkpoints at safe boundaries on SIGTERM; avoid SIGKILL. Verify vLLM synchronization, input/source/config identities and checkpoint state. Stop a broken computation if required to fix it; continue the experiment afterward.

The pipeline is fail-fast and intentionally does NOT automatically retry corrupted stages. Recovery is performed by the heartbeat agent. Do not rerun the pipeline blindly: its exclusive logs and non-overwrite checks will refuse existing stages. Inspect the failed stage, run only required remainder, use --resume only for an existing RL checkpoint with matching config. SFT has no intermediate recovery checkpoint (only a 109-second run); if it must be repeated, use a fresh audited output and do not change an SFT checkpoint already used by an RL arm.

## Comparison and next work

Compare to previous successful PaTR, not directly to the paper for this first stage. Verify baseline raw JSON under results/main100 and supplemental500. Published main100 PaTR success=64.50%, Pass@4=71.00%; supplemental500 success=73.00%, Pass@4=77.80%. Keep these sets separate. One seed is a pilot, not proof of robust improvement.

Analyze success, Pass@4, first-turn legal actions, invalid actions including missing/token-limit outputs, successful path lengths/redundancy, mixed-reward rollout groups, generated tokens, environment actions and wall time including SFT. Preserved raw rollouts permit additional analysis without changing training. Existing golden_diagnostics.py summarizes final retained branches and explicitly notes duplicated tree prefixes; expand it for path redundancy and test diagnostics when needed.

After results, inspect validation curves and propose/run a justified further PaTR experiment in a separate directory. User requests productive continued GPU use, not idle waiting. Do not run purposeless jobs merely to occupy the GPU. Do not repeatedly tune on test results or claim incomparable paper superiority. Preserve all prior experimental archives and baseline weights.
