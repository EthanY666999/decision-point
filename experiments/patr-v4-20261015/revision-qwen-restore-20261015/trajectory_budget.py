"""Pinned rLLM token accumulation, separated from tokenizer-specific rendering."""
from dataclasses import dataclass, field


@dataclass
class TrajectoryBudget:
    limit: int = 10240
    counted_tokens: int = 0
    tokens: list = field(default_factory=list)
    mask: list = field(default_factory=list)

    @property
    def remaining(self):
        return max(0, self.limit - self.counted_tokens)

    def append_turn(self, assistant, assistant_mask, environment, environment_mask, reward, done):
        if len(assistant) != len(assistant_mask) or len(environment) != len(environment_mask):
            raise ValueError('Token/mask length mismatch')
        if any(x not in (0, 1) for x in assistant_mask) or any(environment_mask):
            raise ValueError('Environment tokens must not receive policy gradients')
        if self.remaining == 0:
            raise ValueError('Cannot append to exhausted trajectory')
        previous = self.counted_tokens
        self.counted_tokens += len(assistant) + len(environment)
        truncated = self.counted_tokens >= self.limit
        if truncated:
            keep = self.limit - previous
            ids = (assistant + environment)[:keep]
            masks = (assistant_mask + environment_mask)[:keep]
            if previous + len(assistant) > self.limit:
                reward = 0.0
            done = True
        else:
            ids = assistant if done else assistant + environment
            masks = assistant_mask if done else assistant_mask + environment_mask
        self.tokens.extend(ids)
        self.mask.extend(masks)
        return reward, done, truncated
