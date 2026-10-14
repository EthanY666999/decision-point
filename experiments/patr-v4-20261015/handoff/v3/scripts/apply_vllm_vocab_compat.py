"""Backport vLLM's model-vocabulary input check to the frozen 0.9.2 runtime.

Qwen has model output/embedding rows beyond the tokenizer's largest ID. Keep
those generated IDs intact when a multi-turn rollout sends them back as input.
This changes validation only: no logit mask, tokenizer, or sampling changes.
Upstream: https://docs.vllm.ai/en/latest/api/vllm/renderers/base/
"""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA256 = '5b9dfe8024b354fdcf7e04652735c86d37ba8d7a4d63479b0bb5acc4fa57dbfd'
OLD = '''        if tokenizer is not None:
            max_input_id = max(prompt_ids, default=0)
            if max_input_id > tokenizer.max_token_id:
                raise ValueError(
                    f"Token id {max_input_id} is out of vocabulary")
'''
NEW = '''        if tokenizer is not None:
            max_input_id = max(prompt_ids, default=0)
            min_input_id = min(prompt_ids, default=0)
            # PaTR compatibility backport: Qwen's model vocabulary includes
            # rows beyond the tokenizer. Preserve those generated token IDs.
            model_max_id = model_config.get_vocab_size() - 1
            if min_input_id < 0:
                raise ValueError(
                    f"Token id {min_input_id} is out of vocabulary")
            if max_input_id > max(tokenizer.max_token_id, model_max_id):
                raise ValueError(
                    f"Token id {max_input_id} is out of vocabulary")
'''


def main():
    assert importlib.metadata.version('vllm') == '0.9.2', 'Reassess patch for a different vLLM release'
    target = Path(importlib.util.find_spec('vllm.engine.llm_engine').origin)
    before = target.read_bytes()
    source = before.decode()
    backup = ROOT / 'artifacts/engineering/vllm_vocab_compat/original_llm_engine.py'
    if NEW not in source:
        assert hashlib.sha256(before).hexdigest() == ORIGINAL_SHA256, 'Unexpected dependency source'
        assert source.count(OLD) == 1
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            assert backup.read_bytes() == before
        else:
            backup.write_bytes(before)
        source = source.replace(OLD, NEW, 1)
        compile(source, str(target), 'exec')
        tmp = target.with_suffix('.patr-compat.tmp')
        tmp.write_text(source)
        tmp.replace(target)
    assert source.count(NEW) == 1 and OLD not in source
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == ORIGINAL_SHA256
    audit = dict(unix=time.time(),vllm_version='0.9.2',target=str(target),
                 original_sha256=ORIGINAL_SHA256,
                 patched_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                 backup=str(backup.relative_to(ROOT)),
                 upstream='https://docs.vllm.ai/en/latest/api/vllm/renderers/base/',
                 sampling_changed=False,tokenizer_changed=False,model_changed=False,
                 scope='Accept generated model-vocabulary IDs on reinjection; retain bounds and length checks')
    out = ROOT / 'reports/vllm_vocab_compat_audit.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
