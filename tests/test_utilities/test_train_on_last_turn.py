"""
Tests for train_on_last_turn_only.

When train_on_last_turn_only is True, trajectories are effectively split so that
only the last assistant turn is trained on (action_mask=1). Previous assistant
turns get action_mask=0 (masked out from loss). This tests both the render-level
mask_flags and the tokenized action_mask/labels.
"""

import pytest
import torch

from chat_bricks import get_template
from chat_bricks.utils import tokenize_conversation

TOKENIZER_MODELS = {
    "qwen2.5": "Qwen/Qwen2.5-3B-Instruct",
}

# Multi-turn conversations with multiple assistant turns (for testing last-turn-only masking)
MULTI_TURN_MESSAGES = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I am fine, thank you."},
    {"role": "user", "content": "What is 3 times 5?"},
    {"role": "assistant", "content": "15"},
    {"role": "user", "content": "OK, what is 3 times 6?"},
    {"role": "assistant", "content": "18"},
]

SINGLE_TURN_MESSAGES = [
    {"role": "user", "content": "Hello."},
    {"role": "assistant", "content": "Hi there!"},
]


@pytest.mark.parametrize("template_name", ["qwen2.5"])
def test_render_mask_flags_only_last_assistant_trainable(template_name):
    """With train_on_last_turn_only=True, only the last element that is assistant content has mask_flag=False."""
    template = get_template(template_name)
    prompt, elements, mask_flags = template.render(
        MULTI_TURN_MESSAGES,
        add_generation_prompt=False,
        train_on_last_turn_only=True,
    )
    # mask_flag True = non-assistant (masked), False = assistant (trainable)
    # We expect exactly one False (the last assistant segment)
    print(f"Mask flags: {mask_flags}")
    assert len(elements) == len(mask_flags)
    assistant_indices = [i for i, m in enumerate(mask_flags) if not m]
    assert len(assistant_indices) == 1, (
        f"Expected exactly one trainable (assistant) segment, got {len(assistant_indices)}. "
        f"mask_flags={mask_flags}"
    )


@pytest.mark.parametrize("template_name", ["qwen2.5"])
def test_render_mask_flags_all_assistant_trainable_when_false(template_name):
    """With train_on_last_turn_only=False, every assistant element has mask_flag=False."""
    template = get_template(template_name)
    _, elements, mask_flags = template.render(
        MULTI_TURN_MESSAGES,
        add_generation_prompt=False,
        train_on_last_turn_only=False,
    )
    assistant_indices = [i for i, m in enumerate(mask_flags) if not m]
    assert len(assistant_indices) >= 2, (
        f"Multi-turn conversation should have multiple assistant segments. "
        f"mask_flags={mask_flags}"
    )


@pytest.mark.parametrize("template_name", ["qwen2.5"])
def test_action_mask_only_last_turn_has_ones(template_name):
    """With train_on_last_turn_only=True, action_mask is 1 only for tokens in the last assistant turn."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODELS[template_name], trust_remote_code=True
    )
    inputs_last_only = tokenize_conversation(
        MULTI_TURN_MESSAGES,
        tokenizer,
        template_name,
        max_length=2048,
        add_generation_prompt=False,
        train_on_last_turn_only=True,
        return_tensors="pt",
    )
    action = inputs_last_only["action_mask"].squeeze(0)
    assert action.dim() == 1
    # All assistant content in multi-turn should not all be 1; only last turn is 1
    num_ones = (action == 1).sum().item()
    num_zeros = (action == 0).sum().item()
    assert num_ones >= 1, "At least the last assistant reply should have action_mask=1"
    assert num_zeros >= 1, "Previous turns and user/system should have action_mask=0"


@pytest.mark.parametrize("template_name", ["qwen2.5"])
def test_action_mask_trainable_count_smaller_with_last_turn_only(template_name):
    """Sum of action_mask (trainable tokens) is strictly smaller with train_on_last_turn_only=True for multi-turn."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODELS[template_name], trust_remote_code=True
    )
    inputs_all = tokenize_conversation(
        MULTI_TURN_MESSAGES,
        tokenizer,
        template_name,
        max_length=2048,
        add_generation_prompt=False,
        train_on_last_turn_only=False,
        return_tensors="pt",
    )
    inputs_last = tokenize_conversation(
        MULTI_TURN_MESSAGES,
        tokenizer,
        template_name,
        max_length=2048,
        add_generation_prompt=False,
        train_on_last_turn_only=True,
        return_tensors="pt",
    )
    print(inputs_all["action_mask"])
    print(inputs_last["action_mask"])
    trainable_all = inputs_all["action_mask"].sum().item()
    trainable_last = inputs_last["action_mask"].sum().item()
    assert trainable_last < trainable_all, (
        f"train_on_last_turn_only=True should have fewer trainable tokens: "
        f"last_only={trainable_last} vs all={trainable_all}"
    )


@pytest.mark.parametrize("template_name", ["qwen2.5"])
def test_single_turn_last_only_unchanged(template_name):
    """With a single assistant turn, train_on_last_turn_only=True gives same trainable count as False."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODELS[template_name], trust_remote_code=True
    )
    inputs_all = tokenize_conversation(
        SINGLE_TURN_MESSAGES,
        tokenizer,
        template_name,
        max_length=2048,
        add_generation_prompt=False,
        train_on_last_turn_only=False,
        return_tensors="pt",
    )
    inputs_last = tokenize_conversation(
        SINGLE_TURN_MESSAGES,
        tokenizer,
        template_name,
        max_length=2048,
        add_generation_prompt=False,
        train_on_last_turn_only=True,
        return_tensors="pt",
    )
    assert torch.equal(inputs_all["input_ids"], inputs_last["input_ids"])
    assert torch.equal(inputs_all["action_mask"], inputs_last["action_mask"])
    assert torch.equal(inputs_all["labels"], inputs_last["labels"])


@pytest.mark.parametrize("template_name", ["qwen2.5"])
def test_labels_where_action_mask_zero_are_masked(template_name):
    """Positions with action_mask=0 must have labels=-100 (no loss)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODELS[template_name], trust_remote_code=True
    )
    inputs = tokenize_conversation(
        MULTI_TURN_MESSAGES,
        tokenizer,
        template_name,
        max_length=2048,
        add_generation_prompt=False,
        train_on_last_turn_only=True,
        return_tensors="pt",
    )
    labels = inputs["labels"].squeeze(0)
    action_mask = inputs["action_mask"].squeeze(0)
    assert (labels[action_mask == 0] == -100).all(), (
        "All positions with action_mask=0 should have label -100"
    )


