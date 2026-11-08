# Authorized queue revision — 2026-11-09

S05 remains running from e32ac58 with its existing PID/configuration. Only the idle loader tree changes.

After S05 full verification: F03 single extra evaluation -> S13 noSFT-GRPO4 -> S15 noSFT-UCF-v2 -> H01 base FLH evaluation -> H02 F03 FLH evaluation -> H03 SFT-GRPO4 -> H04 SFT-Random K=2 -> H05 SFT-UCF-v2. The remaining compiled order is unchanged. This prefix requires the preceding verification and stops for manual review instead of bypassing a failed predecessor. Ordinary attempt limits and F03 special journal remain unchanged.

S13 and S15 independently start from original /data/models/Qwen3.5-2B, seed42, 400 steps, max_response_tokens=4096, validation every25 steps. Relative to S03 and S05 respectively, only model, response limit, and run label differ. Frozen Sokoban data and all other algorithm settings are identical.

S14 is not promoted: it remains in the original later P2 backlog, which was not yet fully compiled. S11/S12/S16/S17 and F37/F38 remain explicitly outstanding.

q35_s13_s15_compare.py reads fixed validation only, checks all100 frozen task dicts and four unique lineages, and reports same-step success/pass@4, mixed/all-failure/all-success task counts, mean response tokens per trajectory. A missing peer is null, never substituted with a different step. Training initial-four/all-actual-leaf counts remain separately labelled. The heartbeat invokes this CPU-only report while S13/S15 run; the supervisor also writes it after each completes. No new GPU acceptance is claimed by these CPU checks.
