Prepared only; production source remains frozen.

This supersedes the token/text portion of delivery_candidates_with_token_audit.
The base patch includes selected runtime-source schema4 identity, two DAPO
episode-weighted logging means, idle/reset-failure guards, and official verifier
token/text checks. Evaluation entry reset is a separate conditional patch and
still requires controlled GPU diagnostic evidence plus explicit source release.

The verifier checks original tokenizer/model assets. Standalone decode uses
skip_special_tokens=True, clean_up_tokenization_spaces=False. If it differs from
saved text, the actual installed vLLM incremental decoder is replayed token by
token using the exact saved request prefix, with skip_special_tokens=True and
spaces_between_special_tokens=True. Both modes are counted. No suffix stripping
or text normalization is performed. Historical schema3 did not freeze decoder
source bytes: current vLLM/transformers/tokenizers versions and two decoder source
hashes are recorded, without claiming a retroactive historical byte freeze.

The preserved real incomplete-UTF8 fixture is
reports/fixtures/vllm_utf8_tail_training_case.json. It includes all318 prompt IDs,
17 response IDs and original raw identity. Exact incremental replay equals the
saved output; standalone decode has an extra trailing two replacement characters.
Same-length forged output remains rejected in regression tests.

CPU candidate validation:129 passed in57.45s, using in-memory candidate modules
only, conditional evaluation-reset variant enabled. Actual train.py,
run_integrity.py and scripts/verify_final.py bytes stayed unchanged. Log:
reports/delivery_candidates_incremental_decode_cpu_full.txt.

After source release, adapt pending tests to final chosen reset behavior and
run the complete real files-on-disk CPU suite again. Keep the old study freeze,
schema3 run fingerprints, raw records and checkpoints intact. Delivery gets a
separate source freeze and one real production-entry short smoke.
