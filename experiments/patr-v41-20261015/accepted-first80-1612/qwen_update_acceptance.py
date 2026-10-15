"""Real full-parameter Qwen check of paper trajectory-mean PPO + clipped KL vs selected-token implementation.

Both sides use the same declared BF16 projected forward/FP32 masters. This does
not claim to validate distributed FSDP numerics or the rollout integration.
"""
import ast
import hashlib
import os
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.func import functional_call
from transformers import AutoModelForCausalLM
from loss_v41 import low_var_kl, trajectory_mean
from training_contract import EpochSampler, Counters
import checkpoint_v4
from engine_v4 import selected_logprobs
from numerical import projected_checkpoint

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
                        [151644, 8948, 198, 2610, 525, 264, 11190, 17847, 13, 151645, 198, 151644, 872, 198, 330, 151643]], device='cuda')
    # Shared prefix, variable assistant lengths and intervening observation masks.
    mask = torch.tensor([[0,0,0,0,0,0,0,0,0,1,1,0,0,1,1], [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1]], device='cuda', dtype=torch.float32)
    advantages = torch.tensor([[1.], [-1.]], device='cuda')
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-6, betas=(.9,.999), eps=1e-8, weight_decay=.01, foreach=False) for m in models]
    anchor = None
    losses = []
    for index, model in enumerate(models):
        if index == 1:
            model.train()
            if os.environ.get('PATR_ACCEPTANCE_NO_CHECKPOINT') != '1':
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model._project_forward_bf16 = True
            model.model._gradient_checkpointing_func = projected_checkpoint
            fullmask = torch.nn.functional.pad(mask, (1, 0))
            current = selected_logprobs(model, ids, torch.ones_like(ids), fullmask, gradient=True)
            selected_anchor = anchor[mask.bool()]
            selected_advantage = advantages.expand_as(mask)[mask.bool()]
            ratio = (current-selected_anchor).exp()
            token_policy = -torch.minimum(ratio*selected_advantage, ratio.clamp(.8, 1.28)*selected_advantage)
            loss = trajectory_mean(token_policy + .001*low_var_kl(current, ref_lp[mask.bool()]), [int(n) for n in mask.sum(-1)], 2)
            loss.backward()
            losses.append(float(loss.detach()))
            del current, loss
            gc.collect()
            continue
        with torch.autocast('cuda', dtype=torch.bfloat16):
            projected = {name: p.to(torch.bfloat16) for name, p in model.model.named_parameters()}
            hidden = functional_call(model.model, projected, (), dict(input_ids=ids, use_cache=False)).last_hidden_state
            logits = model.lm_head(hidden[:, :-1]).float()
            lp = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        if anchor is None:
            # Frozen anchor spans both clipped and unclipped policy ratios.
            anchor = lp.detach() + torch.linspace(-.4, .4, lp.shape[1], device='cuda')
            # Both EOS positions must contribute gradient, not sit on clipped branches.
            anchor[:, -1] = lp.detach()[:, -1]
        if index == 0:
            ref_lp = lp.detach() + torch.linspace(-.3,.3,lp.shape[1],device='cuda')
            pad = lambda t: torch.nn.functional.pad(t, (0, 10240 - t.shape[1]))
            loss = ns['compute_policy_loss'](pad(anchor), pad(lp), pad(advantages.expand_as(lp)), pad(mask), cliprange_low=.2, cliprange_high=.28, loss_agg_mode='seq-mean-token-mean')[0]
            delta=(ref_lp-lp).clamp(-20,20)
            kl=(delta.exp()-delta-1).clamp(-10,10)
            loss=loss+.001*((kl*mask).sum(-1)/mask.sum(-1)).mean()
        loss.backward()
        losses.append(float(loss.detach()))
        del hidden, logits, lp, projected, loss
        gc.collect()
    grad_error = 0.
    checked = 0
    failures = []
    for (name, a), (name2, b) in zip(models[0].named_parameters(), models[1].named_parameters()):
        assert name == name2 and a.grad is not None and b.grad is not None
        delta = (a.grad-b.grad).abs()
        failed = delta > (1e-6 + 1e-3*b.grad.abs())
        if failed.any():
            failures.append(dict(name=name, max_absolute=float(delta.max()), reference_max=float(a.grad.abs().max()), candidate_max=float(b.grad.abs().max()), failed_elements=int(failed.sum())))
        grad_error = max(grad_error, float((a.grad-b.grad).abs().max()))
        checked += a.numel()
    if failures:
        report.update(passed=False, losses=losses, failures=failures, max_gradient_error=grad_error, checkpoint_disabled=os.environ.get('PATR_ACCEPTANCE_NO_CHECKPOINT') == '1', stage='actual_selected_logprob_parameter_gradient')
        name = 'qwen_selected_no_checkpoint_failure.json' if report['checkpoint_disabled'] else 'qwen_selected_checkpoint_failure.json'
        (ROOT/'reports'/name).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2), flush=True)
        raise AssertionError('Actual selected-logprob path exceeds declared gradient tolerance')
    norms = [float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)) for m in models]
    assert all(torch.isfinite(torch.tensor(norms)))
    # This fixture compares clipped AdamW math, not production skip policy.
    # Record the production disposition explicitly; do not change its threshold.
    report['fixture_preclip_norms']=norms
    report['production_would_skip_at_norm_10']=[n>=10 for n in norms]
    report['optimizer_fixture_scope']='Unconditional post-clip AdamW equivalence; production skip policy is tested separately, not bypassed in training'
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
    report.update(passed=True, actual_selected_logprob_checkpointed_path=True, losses=losses, compared_parameters=checked, max_gradient_error=grad_error, max_parameter_error=param_error, actual_parameter_change=actual_change, checkpoint_restore_exact=True, checkpoint_bytes=checkpoint_bytes, preclip_norms=norms, seconds=time.time()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    (ROOT/'reports/qwen_update_acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
