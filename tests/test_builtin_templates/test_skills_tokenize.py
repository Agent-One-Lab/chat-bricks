"""Tokenisation tests for skill rendering.

The skills= argument flows through :meth:`Chat.tokenize` → :meth:`Template.encode`
→ ``_encode_standard``. ``test_skills.py`` only checks the textual prompt — these
tests pin the tokenised side: skill text is encoded into ``input_ids``, action_mask
keeps the system block frozen at 0, and ``train_on_last_turn_only=True`` still
restricts trainable tokens to the last assistant turn even when a skills block
inflates the system prompt.
"""

import pytest
import torch
from transformers import AutoTokenizer

from chat_bricks import Chat


TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as e:
        pytest.skip(f"Tokenizer {TOKENIZER_ID} not available: {e}")


MULTI_TURN_MESSAGES = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Add 3 and 5."},
    {"role": "assistant", "content": "8"},
    {"role": "user", "content": "Now multiply by 2."},
    {"role": "assistant", "content": "16"},
]

SINGLE_TURN_MESSAGES = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
]

SKILLS = [
    {"name": "add-numbers", "description": "Adds two integers."},
    {"name": "word-count", "description": "Counts words in text."},
]


def _decode(tokenizer, input_ids):
    return tokenizer.decode(input_ids[0], skip_special_tokens=False)


def test_tokenize_includes_skill_text_in_input_ids(tokenizer):
    """The skill catalogue must round-trip through tokenisation — decoding the
    final input_ids should contain the skill names."""
    chat = Chat("qwen-skills", SINGLE_TURN_MESSAGES, skills=SKILLS)
    inputs = chat.tokenize(tokenizer=tokenizer)
    decoded = _decode(tokenizer, inputs["input_ids"])
    assert "add-numbers" in decoded
    assert "word-count" in decoded
    assert "# Skills" in decoded


def test_tokenize_skills_via_init_kwarg(tokenizer):
    """Passing skills via ``Chat(skills=...)`` should match passing via
    ``tokenize(skills=...)`` directly — they're two routes to the same field."""
    inputs_init = Chat("qwen-skills", SINGLE_TURN_MESSAGES, skills=SKILLS).tokenize(tokenizer)
    inputs_kwarg = Chat("qwen-skills", SINGLE_TURN_MESSAGES).tokenize(tokenizer, skills=SKILLS)
    assert torch.equal(inputs_init["input_ids"], inputs_kwarg["input_ids"])
    assert torch.equal(inputs_init["action_mask"], inputs_kwarg["action_mask"])


def test_tokenize_skills_inflates_prompt_vs_no_skills(tokenizer):
    """A skills-aware template with skills passed should produce strictly more
    tokens than the same call without skills."""
    inputs_no_skills = Chat("qwen-skills", SINGLE_TURN_MESSAGES).tokenize(tokenizer)
    inputs_with_skills = Chat("qwen-skills", SINGLE_TURN_MESSAGES, skills=SKILLS).tokenize(tokenizer)
    assert inputs_with_skills["input_ids"].shape[1] > inputs_no_skills["input_ids"].shape[1]


def test_tokenize_action_mask_excludes_skill_tokens(tokenizer):
    """Skill text lives in the system message, so its tokens must NOT be marked
    trainable in action_mask. Decode the trainable subset and assert no skill
    names leak in."""
    chat = Chat("qwen-skills", SINGLE_TURN_MESSAGES, skills=SKILLS)
    inputs = chat.tokenize(tokenizer)
    input_ids = inputs["input_ids"][0]
    action_mask = inputs["action_mask"][0]
    trainable_ids = input_ids[action_mask.bool()]
    trainable_text = tokenizer.decode(trainable_ids, skip_special_tokens=False)
    assert "add-numbers" not in trainable_text
    assert "word-count" not in trainable_text
    assert "# Skills" not in trainable_text
    # Sanity: trainable text should still contain the assistant's reply
    assert "hello" in trainable_text


def test_tokenize_action_mask_count_unchanged_by_skills(tokenizer):
    """Adding skills inflates the system block but must NOT change the number
    of trainable tokens — those depend on the assistant turns only."""
    inputs_no_skills = Chat("qwen-skills", MULTI_TURN_MESSAGES).tokenize(tokenizer)
    inputs_with_skills = Chat("qwen-skills", MULTI_TURN_MESSAGES, skills=SKILLS).tokenize(tokenizer)
    assert inputs_no_skills["action_mask"].sum() == inputs_with_skills["action_mask"].sum()


def test_train_on_last_turn_only_with_skills(tokenizer):
    """``train_on_last_turn_only=True`` should still keep only the last
    assistant turn trainable when a skills block is present."""
    chat = Chat("qwen-skills", MULTI_TURN_MESSAGES, skills=SKILLS)
    inputs_all = chat.tokenize(tokenizer, train_on_last_turn_only=False)
    inputs_last = chat.tokenize(tokenizer, train_on_last_turn_only=True)

    # Same prompt → same input_ids
    assert torch.equal(inputs_all["input_ids"], inputs_last["input_ids"])
    # Strictly fewer trainable tokens when train_on_last_turn_only=True
    assert inputs_last["action_mask"].sum() < inputs_all["action_mask"].sum()
    # The trainable text under train_on_last_turn_only should be ONLY the final
    # assistant reply ("16") — the earlier "8" must be masked out.
    trainable_text = tokenizer.decode(
        inputs_last["input_ids"][0][inputs_last["action_mask"][0].bool()],
        skip_special_tokens=False,
    )
    assert "16" in trainable_text
    assert "8" not in trainable_text


def test_tokenize_labels_align_with_action_mask(tokenizer):
    """``labels`` should be the input id at trainable positions and -100
    elsewhere — verify the contract holds with skills in the prompt."""
    chat = Chat("qwen-skills", SINGLE_TURN_MESSAGES, skills=SKILLS)
    inputs = chat.tokenize(tokenizer)
    input_ids = inputs["input_ids"][0]
    labels = inputs["labels"][0]
    action_mask = inputs["action_mask"][0]
    assert input_ids.shape == labels.shape == action_mask.shape
    # Wherever action_mask is 1, labels equals input_ids
    assert torch.equal(labels[action_mask.bool()], input_ids[action_mask.bool()])
    # Wherever action_mask is 0, labels is -100
    assert torch.all(labels[~action_mask.bool()] == -100)
