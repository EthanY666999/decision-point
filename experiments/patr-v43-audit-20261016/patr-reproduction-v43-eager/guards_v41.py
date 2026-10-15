"""Persistent monitoring only. Baselines are first 10/20 completed raw batches.

The current observation is tested against previous baseline entries, then appended.
A full baseline is required for relative tests; token-limit is checked immediately.
Empty-learning batches count for length/time, not for gradient baseline.
"""
import math
import statistics
from dataclasses import dataclass, field, asdict

@dataclass
class StabilityGuard:
    lengths: list = field(default_factory=list)
    gradients: list = field(default_factory=list)
    seconds: list = field(default_factory=list)

    def check(self, mean_assistant_tokens, token_limit_fraction, gradient_norms, seconds):
        values=[mean_assistant_tokens,token_limit_fraction,seconds]+list(gradient_norms)
        if any(not math.isfinite(v) or v < 0 for v in values):
            return ['nonfinite_or_negative_monitor']
        reasons=[]
        if token_limit_fraction > .05: reasons.append('token_limit_above_5_percent')
        if len(self.lengths)==10 and mean_assistant_tokens>200 and mean_assistant_tokens>3*statistics.median(self.lengths):
            reasons.append('assistant_length_above_3x')
        if len(self.gradients)==20 and gradient_norms and max(gradient_norms)>20*statistics.median(self.gradients):
            reasons.append('gradient_above_20x')
        if len(self.seconds)==20 and seconds>4*statistics.median(self.seconds):
            reasons.append('step_seconds_above_4x')
        if len(self.lengths)<10:self.lengths.append(mean_assistant_tokens)
        if len(self.seconds)<20:self.seconds.append(seconds)
        if gradient_norms and len(self.gradients)<20:self.gradients.append(max(gradient_norms))
        return reasons

    def state_dict(self):return asdict(self)
