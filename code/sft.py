"""One fixed epoch of action-only Golden Trace SFT; no rollout or test selection."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path


def encode_trace(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    mask = [0] * len(ids)
    actions = 0
    for i, msg in enumerate(messages):
        if msg['role'] != 'assistant':
            continue
        if msg['content'] not in ['```Up```', '```Down```', '```Left```', '```Right```']:
            raise ValueError('Unexpected teacher response')
        prefix = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=True)
        end = tokenizer.apply_chat_template(messages[:i+1], tokenize=True, add_generation_prompt=False)
        assert ids[:len(prefix)] == prefix and ids[:len(end)] == end
        # Train the action and its end-of-turn token, but not the following newline/header.
        target = tokenizer.encode(msg['content'], add_special_tokens=False) + [tokenizer.eos_token_id]
        assert ids[len(prefix):len(prefix)+len(target)] == target
        mask[len(prefix):len(prefix)+len(target)] = [1] * len(target)
        actions += 1
    assert actions and not mask[0]
    return ids, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--optimizer', choices=['adamw', 'adamw8bit'], default='adamw')
    a = p.parse_args()
    from transformers import AutoTokenizer
    from patr_core import atomic_json, append_json
    if a.output.exists():
        raise SystemExit('Refusing to overwrite an existing SFT output')
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    rows = [json.loads(s) for s in a.data.read_text().splitlines()]
    assert len(rows) == 1000 and len({r['task_uid'] for r in rows}) == 1000
    encoded = [encode_trace(tokenizer, r['messages']) for r in rows]
    assert max(len(ids) for ids, _ in encoded) <= 32768
    summary = dict(trajectories=len(rows), target_tokens=sum(sum(m) for _,m in encoded),
                   max_sequence_tokens=max(len(x) for x,_ in encoded),
                   data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
                   seed=42, epochs=1, batch_trajectories=16, micro_batch=1, learning_rate=2e-5,
                   optimizer=a.optimizer,
                   objective='assistant action tokens plus end-of-turn; causal; token mean per batch')
    print(json.dumps(summary), flush=True)
    if a.audit_only:
        return
    from transformers import AutoModelForCausalLM
    import torch
    from train import selected_logprobs
    import fcntl
    lock=(Path(__file__).resolve().parent/'artifacts/.run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another training process holds the workspace lock.')
    a.output.mkdir(parents=True)
    atomic_json(a.output/'config.json', summary)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); random.seed(42)
    torch.set_num_threads(8)
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32,
                     attn_implementation='sdpa', local_files_only=True).to('cuda')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache = False
    model.train()
    if a.optimizer == 'adamw8bit':
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(model.parameters(), lr=2e-5, weight_decay=0.0)
    order = list(range(len(rows))); random.Random(42).shuffle(order)
    for step, off in enumerate(range(0,len(order),16), 1):
        batch = [encoded[i] for i in order[off:off+16]]
        denominator = sum(sum(mask) for _,mask in batch)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for tokens, mask in batch:
            ids = torch.tensor([tokens], device='cuda')
            m = torch.tensor([mask], device='cuda')
            loss = -selected_logprobs(model, ids, torch.ones_like(ids), m, gradient=True).sum()/denominator
            if not torch.isfinite(loss): raise FloatingPointError('SFT loss is not finite')
            loss.backward(); loss_value += loss.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(norm): raise FloatingPointError('SFT gradient is not finite')
        optimizer.step()
        record = dict(step=step, total_steps=63, loss=loss_value, grad_norm=float(norm),
                      elapsed_seconds=time.monotonic()-start, target_tokens=denominator)
        append_json(a.output/'metrics.jsonl',record)
        atomic_json(a.output/'status.json',dict(status='training',**record))
        print(json.dumps(record), flush=True)
    model.config.use_cache=True
    model.save_pretrained(a.output/'final_model',safe_serialization=True)
    tokenizer.save_pretrained(a.output/'final_model')
    summary.update(status='complete',steps=step,elapsed_seconds=time.monotonic()-start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    atomic_json(a.output/'status.json',summary)
    print(json.dumps(summary),flush=True)

if __name__ == '__main__':
    main()
