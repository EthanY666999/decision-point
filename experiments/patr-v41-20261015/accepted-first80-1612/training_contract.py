"""Single-card V4 contract. Not a trainer; acceptance is tracked separately.

Ordering is a local deterministic alternative to upstream distributed balancing.
V4.1 loss averages trajectory means within each actual optimizer minibatch.
"""
import copy
import math
import random
from dataclasses import asdict, dataclass, field

NOMINAL_MINIBATCH = 256
RESPONSE_WIDTH = 10240


def filter_and_advantages(groups, eps=1e-6):
    """Filter uniform binary-outcome groups before unbiased group advantages.

    Return original group/leaf indices to preserve lineage and optimizer ordering.
    No resampling, group truncation, or group statistics across task boundaries.
    """
    rows, rejected = [], []
    for group_id, rewards in enumerate(groups):
        rewards = list(rewards)
        if not rewards or any(not math.isfinite(x) or x not in (0, 1) for x in rewards):
            raise ValueError('Expected nonempty finite binary outcome groups')
        if all(x == 0 for x in rewards) or all(x == 1 for x in rewards):
            rejected.append(group_id)
            continue
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / (len(rewards) - 1))
        rows.extend((group_id, i, (x - mean) / (std + eps)) for i, x in enumerate(rewards))
    return rows, rejected


def trajectory_minibatches(rows):
    for start in range(0, len(rows), NOMINAL_MINIBATCH):
        yield rows[start:start + NOMINAL_MINIBATCH]


def scaled_microbatch_loss(sequence_token_sums):
    """Equivalent to upstream padded-width mean times microbatch-size/256."""
    return sequence_token_sums.sum() / (NOMINAL_MINIBATCH * RESPONSE_WIDTH)


class EpochSampler:
    """Explicit local Python RNG ordering, not claimed identical to DataLoader."""
    def __init__(self, size, seed=42, batch_size=64, drop_last=True):
        if size < batch_size or batch_size <= 0:
            raise ValueError('Dataset must contain at least one full batch')
        self.size, self.batch_size, self.drop_last = size, batch_size, drop_last
        self.rng = random.Random(seed)
        self.epoch, self.cursor = 0, 0
        self.permutation = list(range(size))
        self.rng.shuffle(self.permutation)

    def next_batch(self):
        remaining = self.size - self.cursor
        if remaining == 0 or (self.drop_last and remaining < self.batch_size):
            self.epoch += 1
            self.cursor = 0
            self.permutation = list(range(self.size))
            self.rng.shuffle(self.permutation)
        stop = min(self.cursor + self.batch_size, self.size)
        batch = self.permutation[self.cursor:stop]
        self.cursor = stop
        return batch

    def state_dict(self):
        state = copy.deepcopy(vars(self))
        state['rng'] = self.rng.getstate()
        return state

    def load_state_dict(self, state):
        state = copy.deepcopy(state)
        for key in ('size', 'batch_size', 'drop_last'):
            if state[key] != getattr(self, key):
                raise ValueError('Sampler configuration mismatch: ' + key)
        if sorted(state['permutation']) != list(range(self.size)) or not 0 <= state['cursor'] <= self.size:
            raise ValueError('Corrupt sampler state')
        rng = random.Random()
        rng.setstate(state.pop('rng'))
        self.__dict__.update(state)
        self.rng = rng


@dataclass
class Counters:
    raw_batch_index: int = 0
    accepted_iteration: int = 0
    optimizer_attempts: int = 0
    optimizer_updates: int = 0
    skipped_updates: int = 0
    consecutive_empty: int = 0
    sampled_tasks: int = 0
    sampled_tokens: int = 0
    guard_state: dict = field(default_factory=dict)

    def record_batch(self, tasks, tokens, retained):
        self.raw_batch_index += 1
        self.sampled_tasks += tasks
        self.sampled_tokens += tokens
        self.consecutive_empty = 0 if retained else self.consecutive_empty + 1
        self.accepted_iteration += bool(retained)
        return self.consecutive_empty < 20

    def state_dict(self):
        return asdict(self)


def optimizer_step(optimizer, parameters, counters, torch):
    """Pre-clip norm controls skip; nonfinite is a hard stop, never retried here."""
    counters.optimizer_attempts += 1
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    value = float(norm)
    if not math.isfinite(value):
        counters.skipped_updates += 1
        raise FloatingPointError('Nonfinite gradient norm')
    if value >= 10.0:
        counters.skipped_updates += 1
        return value, False
    optimizer.step()
    counters.optimizer_updates += 1
    return value, True
