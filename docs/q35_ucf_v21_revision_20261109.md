# UCF-v2.1 causal-complement revision — 2026-11-09

This revision implements the user's second protocol definition and the two explicit clarification replies. It supersedes the executable definition in `q35_ucf_v21_20261109.md`; the original failed receipts and source commits remain historical evidence.

## Authorized order and scope

S05 (immutable e32ac58, original PID) → F03's single already-authorized extra evaluation → H01/H02 hard FrozenLake base/SFT evaluations → H03 SFT-GRPO4 → H04 SFT-Random K=2 → H05 corrected SFT-UCF → S13 new-protocol noSFT-GRPO4 → v2.1 one-step GPU acceptance → S15 new-protocol noSFT-UCF. The old suffix follows unchanged, including S14 in its later P2 position.

Only the 80GB Qwen3.5-2B environment is managed here. The user also specified this response protocol for future 9B reasoning-format experiments; that is a portability requirement, not a claim that any 9B job or another machine was modified.

S13/S15 keep the original base Qwen3.5-2B, seed42, reply cap4096, 400 steps and fixed validation every25 steps. Their same-step comparison reports success/pass@4, mixed/all-failure questions and mean reply tokens separately from training samples and final tests.

## New no-SFT response protocol

Future no-SFT entries have `response_protocol=first_complete_action_block`, including H01, S13/S15 and F09/F10. Completed F02 and S01 are not changed or rerun. F02 used the legacy last-block protocol and is **not** a paired control for new noSFT UCF.

The first complete block is an opening triple fence, optional whitespace, an accepted action (case-insensitive Left/Down/Right/Up or the parser's existing1–4 aliases), optional whitespace, and a closing triple fence. Earlier incomplete/invalid fenced text is not mistaken for a complete action block. The new noSFT agent executes the first complete block; the vendor parser remains byte-identical for legacy/SFT runs.

The sampled token sequence and stored old logprobs are sliced after the first sampled token completing the closing fence; they are never retokenized. A closing fence can share its final token with punctuation (for example token27653 is a backtick plus a period). This implementation uses the original request's token-level cutoff: the indivisible completing token is retained and `boundary_token_overhang` records its extra characters; no subsequent token is kept. The user explicitly approved retaining the complete boundary token in the follow-up reply. Later sampled tokens are discarded from the context, trace and training mask, and never executed. If unused later generation hit its length limit, a complete block already present still executes. With no complete block, the usual invalid/length outcome remains. A context-only EOS (mask0, old_logprob0) closes a truncated assistant turn before the unchanged next-observation chat suffix. No synthetic EOS training target is introduced.

Training and evaluation use the same `apply_response` path. vLLM may finish generating an unused suffix before the CPU cutoff; generated-token cost counts that work, while response length and training count retained tokens only. This does not claim a GPU early-stopping optimization.

SFT entries do **not** enable this cutoff. A hypothetical cutoff removes their sampled EOS even when their visible action text is identical, so “all token IDs remain identical after applying cutoff” would be false. The actual disabled SFT path is tested against the historical function using the real2B tokenizer, including actions, text, IDs, masks, old logprobs, rewards and terminal states.

## Node and full class complement

The user explicitly chose **retain the node immediately after the first fence; do not skip whitespace**. Selecting the last whitespace before an observed nonwhite token would condition on a future event. The corrected sampler supports whitespace and other residual tokens, so this lookahead is unnecessary.

The actual sampled reasoning/fence prefix is copied through the earliest token completing the first textual fence, including intrinsic characters in that same token; no later token is inspected or skipped. This handles spaced and fragmented fences, instead of accidentally selecting a later canonical fence token. All action-token variants are scored with unrestricted vLLM `prompt_logprobs` at that exact prefix. No fence probability is multiplied again. Residual mass remains p[0]. The immediate sampled successor defines the original class, with residual tokens in STAY.

Counterfactual sampling excludes the **entire original consequence class**. If it is not STAY, direct `SamplingParams(logit_bias={token:-inf,...})` excludes that class's finite action-token set; every other vocabulary token, including residual/EOS tokens, stays in support. If it is STAY, residual tokens themselves belong to the excluded class; the equivalent allowed list contains exactly the valid action variants whose class is not STAY. Thus pi_cf=1-pi_o and unsampled_mass=0. Empty complement keeps the documented fallback.

The installed vLLM direct constructor and CPU logits processor retain negative infinity exactly. The optional-argument conversion API clamps biases to-100 and is deliberately not used. GPU execution remains separately gated. Reference: https://docs.vllm.ai/en/stable/api/vllm/sampling_params/ .

Every sampled counterfactual X, including an invalid/residual token, is rescored with unrestricted `prompt_logprobs`; the restricted logprob is diagnostic only. EOS stops the forced turn without an extra continuation. PartB starts at X and ends at the retained end of the current turn; copied reasoning/fence tokens have PartB mask0. PartA keeps the original sampled prefix. `cf_parse_mismatch` still records immediate-X label versus executed action; residual whitespace can later produce a valid action, and this is reported rather than silently filtered.

## Question-held-out calibration and historical S05 defect

The original0.5→0.4 counterexample added a later candidate on **the same trajectory/question**, not another question. The unchanged v2 controller used by S05 has this dependency. A64-question reproduction also changes the current rho from0.5 to0.49805447 when only the current question gains one later candidate. S05 is not stopped, patched or restarted.

Future corrected UCF computes c for each question from **all other questions**. No candidate in the held-out question enters its c calibration. Same-question later support leaves the current c/rho unchanged when its current prefix/w and other-trajectory feature values are fixed; another question's changes may alter c. The existing leave-one-trajectory p-hat and baseline continue to use the other three trajectories. Changing those outcomes can legitimately change w; the audit does not confuse that feature dependence with calibration leakage.

The calibration target remains2, rho_min=.03, beta=.25, F=2 and the HT1/rho factor are unchanged. Since the held-out question cannot tune its own c, its expected and realized fork count need not equal2. Actual expected/observed budgets are reported; no full-batch exact budget is claimed. The explicitly named controller-v1 ablation retains its prior-step controller rather than being silently converted to v2 calibration.

The user explicitly approved the corrections for future SFT-UCF too. Those entries select `ucf_corrections=question_loo_complement`, use the same actual-fence conditional scoring and copied-prefix PartB implementation, and report `ucf_version=v2.1-causal-complement`. They are not the old S05 implementation. H05 gets a distinct F03-weight one-step GPU acceptance directory and loading/numerical checks. Its SFT response protocol remains legacy; only future noSFT entries truncate.

Comparability: against historical S05, new noSFT UCF changes initialization, response protocol and implementation; against future corrected SFT-UCF, it changes initialization and response protocol. Neither is a strict “only remove SFT” ablation. S13/S15 share the new protocol and form the appropriate noSFT method comparison.

## Acceptance and evidence

Current receipts live in `/data/runs/queue_q35_hard/ucf_v21_revision_20261109/`. Original failed evidence stays in `ucf_v21_20261109/`; no old acceptance, publication, attempts or failure archive is rewritten.

The corrected reasoning-prefix/fence simulation runs200,000 draws per case. p0=0 has max z1.484089; p0=.08 has max z1.596550, both below4 with zero analytical bias under the stated simulated gradient target. These are CPU mathematical checks, not a claim of GPU or full training acceptance.

Historical S03 has400 batches,102,400 trajectories and676,422 turns. All676,253 complete action blocks have unchanged visible text and action; hypothetical cutoff discards only sampled EOS. S05 through95 has24,320 trajectories and169,239 turns, including169,157 complete blocks with the same property. The first S03 audit only recognized the unspaced fence token and reported58 false endpoint failures; the corrected audit also handles the sampled leading-space closing-fence token, with zero endpoint failures. Both receipts remain available.

The full F02 token audit found297 blocks ending inside an atomic sampled token; those are handled by the documented token-level cutoff and recorded, without retokenization. The earlier strict-character-boundary failure receipt is retained. The historical F02 raw-text survey found319,697 complete action blocks out of334,426 turns. In18,297 turns the first opening fence was not the start of the first complete action block, and609 truncated strings would still confuse the old nonoverlapping last-block parser. This motivates an explicit first-complete-block parser; it does not assert the new mismatch rate is automatically zero or reuse old F02 rewards as new-protocol rewards.

After S13 is fully verified, the gate reads all400 actual new-protocol training batches, verifies frozen task contents and four unique lineages, and checks retained token boundaries and executed actions. The2% threshold is unchanged. The semantic action in the first opened fence (allowing whitespace before the action) is compared against actual executed action; malformed first fenced content maps to invalid. The immediate successor-token diagnostic is separate because newline/space may be residual while the completed block contains an action.

S13 top1 raw logprobs cannot reconstruct every alternative action probability or the held-out entropy weights. The offline audit therefore does not invent an exact expected fork count; the one-step GPU acceptance reports numerical candidate coverage, expected/actual mean forks and cf_parse_mismatch. It cannot run until current CPU gates and the S13 audit pass. Its weights initialize no experiment. F02's old failure is historical, not a gate for the newly authorized protocol.

Default-off legacy UCF source files and the original parser are unchanged; S05 selection records through100 replay byte-for-byte using their stored feature inputs. This verifies default behavior preservation, not that the historical controller was causally correct. General GPU checkpoint-failure recovery remains separately unverified.
