"""Real full-parameter Qwen check of pinned loss vs local single-card reduction.

Both sides use the same declared BF16 projected forward/FP32 masters. This does
not claim to validate distributed FSDP numerics or the rollout integration.
"""
import ast
import hashlib
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.func import functional_call
from transformers import AutoModelForCausalLM
from training_contract import scaled_microbatch_loss
from training_contract import EpochSampler, Counters
import checkpoint_v4

ROOT = Path(__file__).resolve().parent
MODEL = '/home/vipuser/patr-reproduction/artifacts/Qwen2.5-0.5B-Instruct'


def main():
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tree = ast.parse((ROOT / 'reference/upstream_core_algos.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('agg_loss', 'compute_policy_loss')]
    for n in nodes:
        n.decorator_list = []
    ns = {'torch': torch, 'verl_F': SimpleNamespace(masked_mean=lambda v, m: (v*m).sum()/m.sum())}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned_upstream', 'exec'), ns)
    report = dict(scope=__doc__, tolerance=dict(gradient_atol=1e-6, gradient_rtol=1e-3, parameter_atol=2e-7, parameter_rtol=1e-5), passed=False)
    start = time.time()
    models = [AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').cuda().eval() for _ in range(2)]
    ids = torch.tensor([[151644, 8948, 198, 2610, 525, 264, 11190, 17847, 13, 151645, 198, 151644, 872, 198, 510, 151645],
                        [151644, 8948, 198, 2610, 525, 264, 11190, 17847, 13, 151645, 198, 151644, 872, 198, 330, 151645]], device='cuda')
    # Shared prefix, variable assistant lengths and intervening observation masks.
    mask = torch.tensor([[0,0,0,0,0,0,0,0,0,1,1,0,0,1,1], [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1]], device='cuda', dtype=torch.float32)
    advantages = torch.tensor([[1.], [-1.]], device='cuda')
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-6, betas=(.9,.999), eps=1e-8, weight_decay=.01, foreach=False) for m in models]
    anchor = None
    losses = []
    for index, model in enumerate(models):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            projected = {name: p.to(torch.bfloat16) for name, p in model.model.named_parameters()}
            hidden = functional_call(model.model, projected, (), dict(input_ids=ids, use_cache=False)).last_hidden_state
            logits = model.lm_head(hidden[:, :-1]).float()
            lp = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        if anchor is None:
            # Frozen anchor spans both clipped and unclipped policy ratios.
            anchor = lp.detach() + torch.linspace(-.4, .4, lp.shape[1], device='cuda')
        if index == 0:
            pad = lambda t: torch.nn.functional.pad(t, (0, 10240 - t.shape[1]))
            loss = ns['compute_policy_loss'](pad(anchor), pad(lp), pad(advantages.expand_as(lp)), pad(mask), cliprange_low=.2, cliprange_high=.28, loss_agg_mode='seq-mean-token-sum')[0] * (2/256)
        else:
            ratio = (lp-anchor).exp()
            token_loss = -torch.minimum(ratio*advantages, ratio.clamp(.8, 1.28)*advantages)*mask
            loss = scaled_microbatch_loss(token_loss.sum(-1))
        loss.backward()
        losses.append(float(loss.detach()))
        del hidden, logits, lp, projected, loss
        gc.collect()
    grad_error = 0.
    checked = 0
    for (name, a), (name2, b) in zip(models[0].named_parameters(), models[1].named_parameters()):
        assert name == name2 and a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-3, msg=name)
        grad_error = max(grad_error, float((a.grad-b.grad).abs().max()))
        checked += a.numel()
    norms = [float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)) for m in models]
    assert all(torch.isfinite(torch.tensor(norms))) and max(norms) < 10
    before_update = models[0].model.layers[0].self_attn.q_proj.weight.detach().clone()
    for optimizer in optimizers:
        optimizer.step()
    actual_change = float((models[0].model.layers[0].self_attn.q_proj.weight-before_update).abs().max())
    assert actual_change > 0, 'No actual parameter update'
    param_error = 0.
    for (name, a), (_, b) in zip(models[0].named_parameters(), models[1].named_parameters()):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=1e-5, msg=name)
        param_error = max(param_error, float((a-b).abs().max()))
    sampler = EpochSampler(130)
    sampler.next_batch()
    counts = Counters(raw_batch_index=1, accepted_iteration=1, optimizer_attempts=1, optimizer_updates=1, sampled_tasks=64)
    identity = {'acceptance_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'scope': 'acceptance fixture, not formal run fingerprint'}
    path = ROOT/'reports/qwen_acceptance_checkpoint.pt'
    checkpoint_v4.save(path, models[0], optimizers[0], sampler, counts, identity)
    expected_batch = sampler.next_batch()
    expected_random = torch.rand(4, device='cuda')
    with torch.no_grad():
        models[1].model.layers[0].self_attn.q_proj.weight.add_(1.)
    checkpoint_v4.restore(path, models[1], optimizers[1], sampler, counts, identity)
    assert sampler.next_batch() == expected_batch
    torch.testing.assert_close(torch.rand(4, device='cuda'), expected_random, atol=0, rtol=0)
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        for key, value in optimizers[0].state[a].items():
            torch.testing.assert_close(value.cpu(), optimizers[1].state[b][key].cpu(), atol=0, rtol=0)
    assert counts.optimizer_updates == 1
    checkpoint_bytes = path.stat().st_size
    # Disposable acceptance checkpoint only, never a formal training recovery point.
    path.unlink()
    report.update(passed=True, losses=losses, compared_parameters=checked, max_gradient_error=grad_error, max_parameter_error=param_error, actual_parameter_change=actual_change, checkpoint_restore_exact=True, checkpoint_bytes=checkpoint_bytes, preclip_norms=norms, seconds=time.time()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    (ROOT/'reports/qwen_update_acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
