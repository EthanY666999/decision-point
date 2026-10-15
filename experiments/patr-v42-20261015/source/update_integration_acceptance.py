"""Exercise actual update orchestration against pinned reference on a tiny actor."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import update_v4
from training_contract import Counters

ROOT = Path(__file__).resolve().parent


def logprobs(model, ids, attention, mask, gradient=False):
    logits = model(ids[:, :-1])
    return logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)[mask[:, 1:].bool()]


def main():
    torch.manual_seed(42)
    update_v4.selected_logprobs = logprobs
    actor = torch.nn.Embedding(8, 8, dtype=torch.float64, device='cuda')
    expected = copy.deepcopy(actor)
    base = copy.deepcopy(actor).requires_grad_(False)
    with torch.no_grad():base.weight.add_(torch.arange(8,device='cuda')*.03)
    branches = []
    for task in range(45):
        for leaf in range(6):
            ids = [1, 2, 3, 4, 5, (leaf % 3)+1]
            mask = [0, 0, 1, 0, 1, int(leaf % 2 == 0)]
            branches.append(SimpleNamespace(task_id=task, reward=float(leaf%2), ids=ids, mask=mask, ntokens=sum(mask)))
    # A uniform group is rejected, and must never reach optimizer pack.
    branches += [SimpleNamespace(task_id=45, reward=0., ids=[1, 2], mask=[0, 1], ntokens=1) for _ in range(8)]
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6, eps=1e-8, betas=(.9,.999), weight_decay=.01)
    reference_optimizer = torch.optim.AdamW(expected.parameters(), lr=1e-6, eps=1e-8, betas=(.9,.999), weight_decay=.01)
    tree = ast.parse((ROOT/'reference/upstream_core_algos.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('agg_loss','compute_policy_loss')]
    for node in nodes:
        node.decorator_list = []
    ns = dict(torch=torch, verl_F=SimpleNamespace(masked_mean=lambda v,m: (v*m).sum()/m.sum()))
    exec(compile(ast.Module(body=nodes,type_ignores=[]), 'pinned_reference', 'exec'), ns)
    rows = [(b, (b.reward-.5)/((.3)**.5+1e-6)) for b in branches[:270]]
    with torch.no_grad():
        for b, _ in rows:
            ids, attn, mask, _, _ = update_v4.pack([(b, 0.)], 0)
            b.training_old_logprobs = logprobs(expected, ids, attn, mask).tolist()
            b.training_ref_logprobs = logprobs(base, ids, attn, mask).tolist()
    for start in range(0, 270, 256):
        reference_optimizer.zero_grad(set_to_none=True)
        chunk = rows[start:start+256]
        # Independent padded reference, actual trajectory count including short tail.
        cur, old, adv, masks, refs = [], [], [], [], []
        for b, advantage in chunk:
            ids, attn, mask, _, _ = update_v4.pack([(b, 0.)], 0)
            lp = logprobs(expected, ids, attn, mask)
            pad = lambda t: torch.nn.functional.pad(t,(0,10240-len(t)))
            cur.append(pad(lp));old.append(pad(torch.tensor(b.training_old_logprobs,device='cuda')))
            adv.append(pad(torch.full_like(lp,advantage)));masks.append(pad(torch.ones_like(lp)))
            refs.append(pad(torch.tensor(b.training_ref_logprobs,device="cuda")))
        loss = ns['compute_policy_loss'](torch.stack(old),torch.stack(cur),torch.stack(adv),torch.stack(masks),cliprange_low=.2,cliprange_high=.28,loss_agg_mode='token-mean')[0]
        delta=(torch.stack(refs)-torch.stack(cur)).clamp(-20,20)
        kl=(delta.exp()-delta-1).clamp(-10,10)
        padded_mask=torch.stack(masks)
        loss=loss+.001*(kl*padded_mask).sum()/padded_mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expected.parameters(),1.)
        reference_optimizer.step()
    counters = Counters()
    result = update_v4.update(actor,base,optimizer,branches,SimpleNamespace(pad_token_id=0),counters,46)
    assert result['retained_trajectories'] == 270 and result['rejected_groups'] == 1
    assert counters.optimizer_updates == 2
    error = float((actor.weight-expected.weight).abs().max())
    assert error < 1e-9, error
    report = dict(passed=True, retained=270, minibatches=[256,14], uniform_groups_rejected=1, max_parameter_error=error, optimizer_updates=counters.optimizer_updates, scope=__doc__)
    (ROOT/'reports/update_integration_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
