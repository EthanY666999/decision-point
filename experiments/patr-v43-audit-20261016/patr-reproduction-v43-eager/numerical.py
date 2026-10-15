"""Explicit probability and synchronization diagnostics for Qwen2, single GPU."""
import torch
from functools import partial
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint


def projected_checkpoint(function, *args, **kwargs):
    """Keep the projected parameter mapping alive during layer recomputation."""
    bound_args=()
    bound_kwargs={}
    while isinstance(function,partial):
        bound_args=function.args+bound_args
        bound_kwargs={**function.keywords,**bound_kwargs}
        function=function.func
    layer = function.__self__
    params = dict(layer.named_parameters())
    def forward(*inputs):
        return functional_call(layer, params, bound_args+inputs, {**bound_kwargs,**kwargs})
    return checkpoint(forward, *args, use_reentrant=False)


def verify_qwen_weights(actor, engine):
    vm = engine.llm_engine.model_executor.driver_worker.model_runner.model
    checked = []
    def check(name, actual, expected):
        expected = expected.detach().to(actual.dtype)
        if actual.shape != expected.shape or not torch.equal(actual, expected):
            raise AssertionError(f'Inference synchronization mismatch: {name}')
        checked.append(name)
    check('embedding', vm.model.embed_tokens.weight, actor.model.embed_tokens.weight)
    check('lm_head', vm.lm_head.weight, actor.lm_head.weight)
    check('final_norm', vm.model.norm.weight, actor.model.norm.weight)
    for i, (a, v) in enumerate(zip(actor.model.layers, vm.model.layers)):
        for suffix in ('weight', 'bias'):
            parts = [getattr(getattr(a.self_attn, p + '_proj'), suffix, None) for p in ('q', 'k', 'v')]
            if all(p is not None for p in parts):
                check(f'layer{i}.qkv.{suffix}', getattr(v.self_attn.qkv_proj, suffix), torch.cat(parts))
        check(f'layer{i}.o', v.self_attn.o_proj.weight, a.self_attn.o_proj.weight)
        check(f'layer{i}.gate_up', v.mlp.gate_up_proj.weight, torch.cat([a.mlp.gate_proj.weight, a.mlp.up_proj.weight]))
        check(f'layer{i}.down', v.mlp.down_proj.weight, a.mlp.down_proj.weight)
        check(f'layer{i}.input_norm', v.input_layernorm.weight, a.input_layernorm.weight)
        check(f'layer{i}.post_norm', v.post_attention_layernorm.weight, a.post_attention_layernorm.weight)
    if len(actor.model.layers) != len(vm.model.layers):
        raise AssertionError('Layer count mismatch')
    return {'checked_modules': len(checked), 'all_equal': True}


def distribution(values):
    x = values.detach().float().flatten().cpu()
    if not x.numel():
        return {'n': 0}
    return {'n': x.numel(), 'mean': x.mean().item(), 'min': x.min().item(), 'max': x.max().item(),
            'p01': x.quantile(.01).item(), 'p50': x.quantile(.5).item(), 'p95': x.quantile(.95).item(),
            'p99': x.quantile(.99).item(), 'finite': bool(torch.isfinite(x).all())}
