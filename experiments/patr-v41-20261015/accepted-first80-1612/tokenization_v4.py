"""Pinned Qwen parser with disable_thinking=False, as engine default.

The upstream helper removes all assistant-header occurrences using replace;
retain this behavior explicitly, even for a header inside model-generated text.
"""
ASSISTANT = '<|im_start|>assistant\n'


def encode_message(message, tokenizer, generation=False):
    role = message['role']
    if role not in ('system', 'user', 'assistant'):
        raise ValueError('FrozenLake supports system/user/assistant only')
    text = '<|im_start|>' + role + '\n' + message['content'] + '<|im_end|>\n'
    if generation:
        text += ASSISTANT
    if role == 'assistant':
        text = text.replace(ASSISTANT, '')
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens, [int(role == 'assistant')] * len(tokens)
