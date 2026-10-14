"""Regression for model-generated Qwen IDs rejected on the next environment turn."""
from types import SimpleNamespace

import pytest
from vllm.engine.llm_engine import LLMEngine


def validator():
    model = SimpleNamespace(get_vocab_size=lambda: 151936,
                            max_model_len=32768, is_multimodal_model=False)
    return SimpleNamespace(model_config=model,
                           tokenizer=SimpleNamespace(get_lora_tokenizer=lambda _: SimpleNamespace(max_token_id=151664)))


@pytest.mark.parametrize('ids', [[151664], [151725], [151935], [1,151725,151645]])
def test_model_vocab_tokens_can_be_reinjected(ids):
    LLMEngine._validate_model_input(validator(), {'type':'token','prompt_token_ids':ids}, None, prompt_type='decoder')


@pytest.mark.parametrize('ids,reason', [([-1], 'out of vocabulary'),
    ([151936], 'out of vocabulary'), ([], 'empty'), ([1]*32769, 'maximum model length')])
def test_real_bounds_and_context_checks_remain(ids, reason):
    with pytest.raises(ValueError, match=reason):
        LLMEngine._validate_model_input(validator(), {'type':'token','prompt_token_ids':ids}, None, prompt_type='decoder')
