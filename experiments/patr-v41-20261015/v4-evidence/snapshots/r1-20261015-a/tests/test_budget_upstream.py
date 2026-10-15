"""Execute the actual pinned accumulation block, including its break branches."""
import sys
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trajectory_budget import TrajectoryBudget


class UpstreamBudgetTests(unittest.TestCase):
    def test_exact_reference_boundary_matrix(self):
        root = Path(__file__).resolve().parents[1]
        source = (root/'reference/upstream_execution.py').read_text()
        start = source.index('            response_token_len += len(assistant_msg_tokens)')
        stop = source.index('            if step_idx == self.max_steps - 1:', start)
        code = 'while True:\n' + textwrap.indent(textwrap.dedent(source[start:stop]), '    ') + '\n    break\n'
        checks = 0
        for prior in (0, 3, 7):
            for assistant_length in (1, 3, 7, 10):
                for env_length in (0, 2, 8):
                    for done in (False, True):
                        assistant, env = list(range(assistant_length)), list(range(100, 100+env_length))
                        step = SimpleNamespace(reward=1., done=done)
                        ns = dict(response_token_len=prior, response_tokens=[9]*prior, response_masks=[0]*prior,
                                  assistant_msg_tokens=assistant, assistant_msg_masks=[1]*assistant_length,
                                  env_msg_tokens=env, env_msg_masks=[0]*env_length,
                                  self=SimpleNamespace(enforce_max_prompt_length=False, max_response_length=10, trajectory_timeout=100),
                                  agent=SimpleNamespace(get_current_state=lambda: step), cur_step=step,
                                  done=done, next_observation=None, total_time=0, termination_reason=None)
                        exec(code, ns)
                        b = TrajectoryBudget(10, prior, [9]*prior, [0]*prior)
                        reward, terminal, truncated = b.append_turn(assistant, [1]*assistant_length, env, [0]*env_length, 1., done)
                        self.assertEqual((b.counted_tokens, b.tokens, b.mask, reward), (ns['response_token_len'], ns['response_tokens'], ns['response_masks'], step.reward))
                        self.assertEqual(truncated, ns['termination_reason'] == 'TRUNCATION')
                        self.assertEqual(terminal, done or truncated)
                        checks += 1
        self.assertEqual(checks, 72)
