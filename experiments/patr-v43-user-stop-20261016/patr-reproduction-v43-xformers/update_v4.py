"""Whole-task outcome advantages, fixed-anchor trajectory-minibatch PPO.

v4.2: minibatch token-mean aggregation (verl default / DAPO)."""
import torch
from loss_v41 import low_var_kl, trajectory_mean, token_mean
from engine_v4 import selected_logprobs
from training_contract import filter_and_advantages, trajectory_minibatches, optimizer_step


def micros(rows, max_rows=2, token_limit=6000):
    # Stable length ordering only inside an optimizer minibatch; disclosed local order.
    batch = []
    for row in sorted(rows, key=lambda row: len(row[0].ids)):
        if batch and (len(batch) >= max_rows or (len(batch)+1)*len(row[0].ids) > token_limit):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


def pack(rows, pad, with_anchor=False):
    width = max(len(b.ids) for b, _ in rows)
    ids = torch.full((len(rows), width), pad, device='cuda', dtype=torch.long)
    attention = torch.zeros_like(ids)
    mask = torch.zeros_like(ids)
    advantages, anchors = [], []
    for i, (b, advantage) in enumerate(rows):
        n = len(b.ids)
        ids[i, :n] = torch.tensor(b.ids, device='cuda')
        attention[i, :n] = 1
        mask[i, :n] = torch.tensor(b.mask, device='cuda')
        advantages.extend([advantage]*b.ntokens)
        if with_anchor:
            assert len(b.training_old_logprobs) == b.ntokens
            anchors.extend(b.training_old_logprobs)
    return ids, attention, mask, torch.tensor(advantages, device='cuda'), torch.tensor(anchors, device='cuda')


def update(actor, reference, optimizer, branches, tokenizer, counters, n_tasks):
    groups = [[b for b in branches if b.task_id == i] for i in range(n_tasks)]
    indices, rejected = filter_and_advantages([[b.reward for b in group] for group in groups])
    rows = [(groups[g][j], advantage) for g, j, advantage in indices]
    outcomes = [[b.reward for b in group] for group in groups]
    # v4.3: "rejected" = zero-variance groups; they are KEPT with advantage 0 (no filter).
    report = dict(rejected_groups=len(rejected), accepted_groups=n_tasks-len(rejected), retained_trajectories=len(rows),
        zero_variance_all_fail=sum(all(r == 0 for r in g) for g in outcomes),
        zero_variance_all_success=sum(all(r == 1 for r in g) for g in outcomes),
        informative_trajectories=sum(len(g) for g in outcomes if not (all(r == 0 for r in g) or all(r == 1 for r in g))), optimizer_updates=0, optimizer_skips=0, gradient_norms=[], loss=0., reference_sampled_logratio_sum=0., reference_sampled_tokens=0,ratio_min=None,ratio_max=None,ratio_sum=0.,ratio_tokens=0,clipped_tokens=0,anchor_logratio_sum=0.)
    report['eos_advantage_mass'] = {str(t): {'positive': 0., 'negative': 0.} for t in (151645, 151643)}
    for branch, advantage in rows:
        for token, selected in zip(branch.ids, branch.mask):
            if selected and str(token) in report['eos_advantage_mass']:
                report['eos_advantage_mass'][str(token)]['positive' if advantage > 0 else 'negative'] += abs(advantage)
    if not rows:
        return report
    actor.train()
    # All anchors are frozen before any optimizer update, including later tails.
    with torch.no_grad():
        for micro in micros(rows):
            ids, attention, mask, _, _ = pack(micro, tokenizer.pad_token_id)
            anchor = selected_logprobs(actor, ids, attention, mask)
            ref = selected_logprobs(reference, ids, attention, mask)
            if not torch.isfinite(anchor).all() or not torch.isfinite(ref).all():
                raise FloatingPointError('Nonfinite frozen training anchor/reference')
            report['reference_sampled_logratio_sum'] += float((anchor-ref).sum())
            report['reference_sampled_tokens'] += anchor.numel()
            offset = 0
            for branch, _ in micro:
                branch.training_old_logprobs = anchor[offset:offset+branch.ntokens].cpu().tolist()
                branch.training_ref_logprobs = ref[offset:offset+branch.ntokens].cpu().tolist()
                offset += branch.ntokens
    for minibatch in trajectory_minibatches(rows):
        optimizer.zero_grad(set_to_none=True)
        minibatch_tokens = sum(b.ntokens for b, _ in minibatch)
        for micro in micros(minibatch):
            ids, attention, mask, advantages, anchor = pack(micro, tokenizer.pad_token_id, True)
            current = selected_logprobs(actor, ids, attention, mask, gradient=True)
            if not torch.isfinite(current).all():
                raise FloatingPointError('Nonfinite current policy logprob')
            ratio = torch.exp(current-anchor)
            if not torch.isfinite(ratio).all():
                raise FloatingPointError('Nonfinite policy ratio')
            detached = ratio.detach()
            lo,hi = float(detached.min()),float(detached.max())
            report['ratio_min'] = lo if report['ratio_min'] is None else min(lo,report['ratio_min'])
            report['ratio_max'] = hi if report['ratio_max'] is None else max(hi,report['ratio_max'])
            report['ratio_sum'] += float(detached.sum())
            report['ratio_tokens'] += detached.numel()
            report['clipped_tokens'] += int(((detached<.8)|(detached>1.28)).sum())
            report['anchor_logratio_sum'] += float((current.detach()-anchor).sum())
            ref_logp = torch.tensor([v for b, _ in micro for v in b.training_ref_logprobs], device=current.device, dtype=current.dtype)
            kl = low_var_kl(current, ref_logp)
            policy = -torch.minimum(ratio*advantages, ratio.clamp(.8, 1.28)*advantages)
            # v4.2: token-mean replaces v4.1 Eq.2 trajectory mean (fence-loop evidence, batches 55-58).
            loss = token_mean(policy + .001*kl, minibatch_tokens)
            report['low_var_kl_token_sum'] = report.get('low_var_kl_token_sum', 0.) + float(kl.detach().sum())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PPO loss')
            loss.backward()
            report['loss'] += float(loss.detach())
        norm, did_update = optimizer_step(optimizer, actor.parameters(), counters, torch)
        report['gradient_norms'].append(norm)
        report['optimizer_updates'] += int(did_update)
        report['optimizer_skips'] += int(not did_update)
    optimizer.zero_grad(set_to_none=True)
    report['reference_sampled_logratio_mean'] = report['reference_sampled_logratio_sum']/report['reference_sampled_tokens']
    report['ratio_mean'] = report['ratio_sum']/report['ratio_tokens']
    report['clip_fraction'] = report['clipped_tokens']/report['ratio_tokens']
    report['anchor_sampled_logratio_mean'] = report['anchor_logratio_sum']/report['ratio_tokens']
    report['reference_diagnostic_definition'] = 'sampled log p_actor minus log p_base on retained canonical assistant tokens; not exact categorical KL'
    return report
