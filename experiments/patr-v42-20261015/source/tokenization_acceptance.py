import ast
import json
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoTokenizer
from tokenization_v4 import encode_message
from patr_core import start_branch, apply_response

ROOT = Path(__file__).resolve().parent


def main():
    tokenizer = AutoTokenizer.from_pretrained('/home/vipuser/patr-reproduction/artifacts/Qwen2.5-0.5B-Instruct', local_files_only=True)
    tree = ast.parse((ROOT/'reference/upstream_parser.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('ChatTemplateParser', 'QwenChatTemplateParser')]
    ns = {'PreTrainedTokenizerBase': object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned_parser', 'exec'), ns)
    tree = ast.parse((ROOT/'reference/upstream_agent_utils.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'convert_messages_to_tokens_and_masks']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned_tokenizer_helper', 'exec'), ns)
    parser = ns['QwenChatTemplateParser'](tokenizer, disable_thinking=False)
    checks = 0
    for role in ('system', 'user', 'assistant'):
        for text in ('', '```Right```', '中文\nThought: go left. ```Left```', '<|im_start|>assistant\n literal header'):
            for generation in (False, True):
                message = dict(role=role, content=text)
                expected = ns['convert_messages_to_tokens_and_masks']([message], tokenizer, parser, contains_generation_msg=generation)
                assert encode_message(message, tokenizer, generation) == expected
                checks += 1
    cfg = dict(max_turns=10, max_response_tokens=10240, is_slippery=False, prompt_profile='legacy')
    task = dict(size=2, p=.8, seed=42, desc=['SG', 'FF'], generation_mode='legacy', generation_max_steps=8)
    branch = start_branch(task, 0, tokenizer, cfg, '0')
    initial, masks = ns['convert_messages_to_tokens_and_masks'](branch.agent.chat_completions, tokenizer, parser, contains_first_msg=True, contains_generation_msg=True)
    assert branch.ids == initial and branch.mask == masks
    text = '```Right```'
    raw = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    output = SimpleNamespace(token_ids=raw, logprobs=[{t: SimpleNamespace(logprob=-.1)} for t in raw], text=text, finish_reason='stop')
    apply_response(branch, output, tokenizer, cfg, sampling_seed=42)
    assistant, assistant_mask = ns['convert_messages_to_tokens_and_masks']([branch.agent.chat_completions[-2]], tokenizer, parser)
    env, _ = ns['convert_messages_to_tokens_and_masks']([branch.agent.chat_completions[-1]], tokenizer, parser, contains_generation_msg=True)
    assert branch.ids == initial + assistant and branch.mask == masks + assistant_mask
    assert branch.budget.counted_tokens == len(assistant) + len(env)
    assert branch.reward == 1 and branch.terminal == 'success'
    report = dict(passed=True, message_cases=checks, success_rollout_integration=True, sampled_assistant_tokens=len(raw), canonical_assistant_tokens=len(assistant), terminal_environment_budget_tokens=len(env), scope='Real tokenizer and synthetic sampler success fixture, not vLLM generated rollout')
    (ROOT/'reports/tokenization_acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
