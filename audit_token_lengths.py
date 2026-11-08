"""Optional CPU-only token length audit; requires transformers and a local tokenizer."""
import argparse
import collections.abc
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('release')
    parser.add_argument('tokenizer')
    args = parser.parse_args()
    root = Path(args.release)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    lengths = []
    with gzip.open(root / 'sft_messages.jsonl.gz', 'rt') as f:
        for line in f:
            row = json.loads(line)
            result = tok.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=False, enable_thinking=False)
            ids = result['input_ids'] if isinstance(result, collections.abc.Mapping) else result
            if ids and isinstance(ids[0], list):
                assert len(ids) == 1
                ids = ids[0]
            assert isinstance(ids, list) and ids and isinstance(ids[0], int)
            lengths.append(len(ids))
    ordered = sorted(lengths)
    report = dict(n=len(lengths), model='Qwen3.5-2B', tokenizer_path=args.tokenizer,
                  transformers_version=importlib.metadata.version('transformers'), enable_thinking=False,
                  mean_tokens=sum(lengths)/len(lengths), max_tokens=max(lengths), min_tokens=min(lengths),
                  p50=ordered[len(ordered)//2], p90=ordered[int(.9*(len(ordered)-1))],
                  above_2048=sum(n>2048 for n in lengths), above_4096=sum(n>4096 for n in lengths),
                  above_8192=sum(n>8192 for n in lengths),
                  chat_template_sha256=hashlib.sha256(str(tok.chat_template).encode()).hexdigest(),
                  note='Full episodes, native chat template, no generation prompt. No truncation or training performed.')
    (root / 'token_lengths_qwen35_2b.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
