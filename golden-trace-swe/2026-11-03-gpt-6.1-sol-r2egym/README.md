# SWE Golden Trace: GPT-6.1, newly executed R2E-Gym training trajectories

**Published verified unique trajectories: 32 / 1,000 target. Collection is in progress.**

These are SWE repair training demonstrations for Qwen3-4B-Instruct-2507 SFT, separately namespaced from FrozenLake `golden-trace/`. They are not SWE-Bench Verified evaluation results. Teacher: Codex CLI requested model `gpt-6.1-sol`, medium reasoning; the CLI does not independently expose the provider serving-model ID. No model fallback. Data is newly generated through real Docker actions, not copied from public teacher completions.

Training metadata: `zhenghaoxu/R2E-Gym-Lite-with-Difficulty`, revision `87fb43eb4d6c71b1e86292346e45f52b71ea7691`, split `filtered_1_7`. Seed-42 candidate order followed by success filtering. Direct overlap with SWE-Bench Verified is excluded by instance ID, normalized issue text and commit hash; this is not a semantic or pretraining leakage guarantee.

## Acceptance

Each task requires a failing original baseline, a passing reference-source-patch calibration, explicit teacher submission, a nonempty permitted source patch, and independent reapplication/testing in a fresh container. Gold patches and private grading metadata are excluded from teacher prompts and this dataset. The pinned R2E parser compares **all expected test statuses**, including preexisting failures; acceptance does not mean every test is green. `verification.jsonl` reports passing/nonpassing counts and patch/log hashes. Full raw verification logs and teacher invocation records are retained on the remote server.

The Docker adapter uses R2E-Gym source revision `0d94c4eb9431cd195c55a7ea3abd54006c9a1735`; this is an adapter with the upstream status comparison and a nonempty-test check, not an unmodified upstream agent run. Containers have network disabled and no host mounts or Docker socket. Cases whose own reference cannot be certified are recorded as environment rejections, never positive demonstrations.

## Files and usage

- `trajectories/part-*.jsonl`: full accepted episodes, actions, real observations, patch and verification summary; at most 10 trajectories per shard.
- `sft/part-*.jsonl`: observable conversations for supervised training, no hidden chain-of-thought. Concatenate shards once; do not also include `trajectories/` as duplicate SFT examples.
- Action contract: `<function=bash><parameter=command>...</parameter></function>` or `<function=submit></function>`. Train/evaluate with this exact contract or an explicitly tested tool-format conversion. User observations must be masked from the SFT loss.
- `manifest.json`: file hashes, target and accepted counts. Current partial shard may grow; completed shards remain stable.
- `generator/`: generation and relay source. Runner and all durable records live on the remote server; only the authenticated Codex teacher relay and transient response buffers run on the Mac. No account credentials are included. A Mac disconnect pauses teacher delivery; it does not create synthetic observations.

Updates go to **main**. Existing FrozenLake files are unchanged. The 1,000 target is not guaranteed: only unique independently accepted tasks count. No Qwen SFT or PaTR performance improvement has been measured by this collection itself.
