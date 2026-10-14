"""Explicit reproduction corrections and separately labelled behavioral variants.

These are local protocol choices, not claims about unpublished author settings.
The inherited parser is intentionally unchanged.
"""
import sys
from pathlib import Path

_VENDOR = str(Path(__file__).resolve().parent / 'vendor/rllm_source')
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
from rllm.agents.frozenlake_agent import FrozenLakeAgent


FORMAT_REMINDER = (
    "In every turn, put exactly one next action in a complete triple-backtick "
    "block: ```Up```, ```Down```, ```Left```, or ```Right```. "
    "P and G are map symbols, not actions."
)


def profile_config(name):
    if name == "legacy":
        return dict(protocol_profile=name, prompt_profile="legacy",
                    invalid_action_penalty=0.0, loss_normalization="trajectory", dapo_loss_normalization="global_token",
                    repeat_check_every_turn=False, patr_tie_break="stable",
                    tree_random_process_pruning=True,
                    arpo_vocabulary_policy="full_model_vocab")
    if name not in ("corrected_v2", "behavior_variant_v2"):
        raise ValueError(f"Unknown protocol profile: {name}")
    result = dict(protocol_profile=name, prompt_profile="corrected_v2",
                  invalid_action_penalty=0.0, loss_normalization="trajectory", dapo_loss_normalization="global_token",
                  repeat_check_every_turn=True, patr_tie_break="random",
                  tree_random_process_pruning=False,
                  arpo_vocabulary_policy="tokenizer_vocab")
    if name == "behavior_variant_v2":
        result.update(invalid_action_penalty=0.05, loss_normalization="group_token", dapo_loss_normalization="group_token")
    return result


class CorrectedFrozenLakeAgent(FrozenLakeAgent):
    """Accurate dynamics and uniform format feedback; same inherited strict parser."""

    def __init__(self, max_steps, is_slippery=False):
        super().__init__(max_steps=max_steps)
        dynamics = ("Moves can slip perpendicular to the chosen direction."
                    if is_slippery else
                    "Movement is deterministic: there is no slipping. A move beyond "
                    "the map boundary leaves your position unchanged.")
        self.messages[0]["content"] = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
            "You are walking on a frozen lake.\n"
            "Goal: move player P onto G. Symbols: _ frozen tile; O hole; "
            "G goal; P player. Avoid holes. " + dynamics + "\n"
            "Reaching G gives outcome reward +1. A hole or failure gives 0.\n"
            "Plan a safe path with as few moves as possible. Briefly explain the "
            "player and goal positions and why the next move is safe, then give "
            "the action. " + FORMAT_REMINDER
        )

    def update_from_env(self, observation, reward, done, info, **kwargs):
        obs = str(observation)
        message = (f"Current Observation ({self.step}):\n{obs}\n"
                   "P has not reached G yet. Please give the next action.\n" + FORMAT_REMINDER)
        previous_action = info.get("parsed_action")
        if previous_action == 0:
            message += "\nThe previous response did not contain a valid action in the required format."
        elif previous_action is not None and info.get("position_unchanged"):
            message += ("\nThe previous action was valid, but the position did not change "
                        "because movement was blocked by the map boundary.")
        if self.max_steps is not None:
            message += f"\nThe maximum number of steps remaining is {max(0, self.max_steps-self.step)}."
        self.messages.append({"role": "user", "content": message})
        self.current_observation = obs


def format_status(text, parsed_action):
    if parsed_action in (1, 2, 3, 4):
        return "valid"
    if text.count("```") < 2:
        return "missing_or_incomplete_fence"
    return "invalid_action_content"
