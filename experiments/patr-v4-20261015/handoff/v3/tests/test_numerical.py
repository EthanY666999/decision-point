"""CPU regression for temporary bf16 parameters surviving checkpoint replay."""
import copy
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.func import functional_call

from numerical import projected_checkpoint


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.gain = nn.Parameter(torch.linspace(.5, 1.2, 8))
        self.calls = 0

    def forward(self, hidden, residual=None, *, scale=1., shift=0.):
        self.calls += 1
        value = hidden if residual is None else hidden + residual
        return torch.tanh(self.linear(value) * self.gain * scale + shift)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(3)])

    def forward(self, hidden, mode, checkpointed):
        for i, layer in enumerate(self.layers):
            kwargs = dict(shift=.125)
            if mode == 'bound_method':
                function = layer.__call__
            elif mode == 'partial_kwargs':
                # Transformers/Qwen passes a partial around the bound method.
                # Call-time kwargs must override values already bound in partial.
                function = partial(layer.__call__, scale=.75+i/8, shift=-10.)
            elif mode == 'partial_args_and_kwargs':
                prefix = torch.full_like(hidden, .125)
                function = partial(partial(layer.__call__, prefix, scale=.5, shift=-20.),
                                   scale=.75+i/8, shift=-10.)
            else:
                raise ValueError(mode)
            hidden = (projected_checkpoint(function, hidden, **kwargs)
                      if checkpointed else function(hidden, **kwargs))
        return hidden


@pytest.mark.parametrize('mode', ['bound_method', 'partial_kwargs', 'partial_args_and_kwargs'])
def test_multilayer_projected_checkpoint_matches_native_bf16_outputs_and_all_gradients(mode):
    torch.manual_seed(20261014)
    model = Model()
    native = copy.deepcopy(model).to(torch.bfloat16)
    hidden = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)/24
    original_parameters = dict(model.named_parameters())
    projected = {name: parameter.to(torch.bfloat16) for name, parameter in original_parameters.items()}
    actual = functional_call(model, projected, (hidden, mode, True))
    expected = native(hidden, mode, False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    native_parameters = dict(native.named_parameters())
    for name, parameter in model.named_parameters():
        assert parameter is original_parameters[name]
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0, f'Broken gradient path: {name}'
        torch.testing.assert_close(parameter.grad, native_parameters[name].grad.float(), rtol=0, atol=0)
    assert all(layer.calls >= 2 for layer in model.layers), 'Checkpoint replay was not exercised'
    assert all(layer.calls == 1 for layer in native.layers)


def test_partial_kwarg_override_is_not_discarded_during_recomputation():
    torch.manual_seed(7)
    model = Model()
    native = copy.deepcopy(model).to(torch.bfloat16)
    hidden = torch.linspace(-1, 1, 24, dtype=torch.bfloat16).reshape(3, 8)
    projected = {name: p.to(torch.bfloat16) for name, p in model.named_parameters()}
    output = functional_call(model, projected, (hidden, 'partial_args_and_kwargs', True))
    target = native(hidden, 'partial_args_and_kwargs', False)
    # Nonuniform downstream derivative catches a recomputation using wrong kwargs.
    weights = torch.arange(1, 25).reshape(3, 8).float()
    (output.float()*weights).sum().backward()
    (target.float()*weights).sum().backward()
    for (name, parameter), (_, reference) in zip(model.named_parameters(), native.named_parameters()):
        torch.testing.assert_close(parameter.grad, reference.grad.float(), rtol=0, atol=0, msg=name)


def test_policy_logits_allowed_vocabulary_has_finite_kl_and_zero_forbidden_gradients():
    from train import policy_logits

    class FourColumnHead(nn.Module):
        def __init__(self, values):
            super().__init__()
            self.logits = nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, hidden):
            return self.logits

    actor_head = FourColumnHead([[1., 10000., -1., 5000.], [.5, -10000., 2., 10000.]])
    ref_head = FourColumnHead([[-.5, -9000., 1.5, 9000.], [2., 9000., -.5, -9000.]])
    actor = SimpleNamespace(lm_head=actor_head, _allowed_token_ids=[0, 2])
    reference = SimpleNamespace(lm_head=ref_head, _allowed_token_ids=[0, 2])
    dummy_hidden = torch.zeros(2, 1)
    masked = policy_logits(actor, dummy_hidden)
    logp = torch.log_softmax(masked, -1)
    probability = logp.exp()
    assert torch.equal(probability[:, [1, 3]], torch.zeros(2, 2))
    torch.testing.assert_close(probability[:, [0, 2]],
        torch.softmax(actor_head.logits[:, [0, 2]], -1), rtol=1e-6, atol=0)
    same_logq = torch.log_softmax(policy_logits(actor, dummy_hidden).detach(), -1)
    same_kl = (probability * (logp - same_logq)).sum(-1)
    assert torch.isfinite(same_kl).all() and torch.equal(same_kl, torch.zeros(2))
    with torch.no_grad():
        logq = torch.log_softmax(policy_logits(reference, dummy_hidden), -1)
    kl = (probability * (logp - logq)).sum(-1)
    assert torch.isfinite(kl).all() and (kl > 0).all()
    kl.sum().backward()
    gradient = actor_head.logits.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.equal(gradient[:, [1, 3]], torch.zeros(2, 2))
    assert (gradient[:, [0, 2]].abs() > 0).all()
    assert ref_head.logits.grad is None
