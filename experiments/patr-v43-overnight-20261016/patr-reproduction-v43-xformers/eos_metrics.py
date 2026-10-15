"""Metrics-only EOS union; never changes sampling or training token IDs."""
import json
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=8)
def configured_eos_ids(model_path):
    config=json.loads((Path(model_path)/'generation_config.json').read_text())
    value=config.get('eos_token_id',[])
    if isinstance(value,int):value=[value]
    if not isinstance(value,list) or any(type(v) is not int or v<0 for v in value):
        raise ValueError('Invalid model EOS configuration')
    return frozenset(value)

def effective_eos_ids(tokenizer, cfg):
    return configured_eos_ids(cfg['model']) | {tokenizer.eos_token_id}

def response_metrics(tokens, eos_ids):
    count=sum(t in eos_ids for t in tokens)
    return dict(response_tokens_excluding_eos=len(tokens)-count,
                sampled_eos_count=count,
                sampled_terminal_eos_id=tokens[-1] if tokens and tokens[-1] in eos_ids else None)
