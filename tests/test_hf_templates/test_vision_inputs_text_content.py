"""Text-only content must yield no vision inputs, whatever its format.

HF templates for text-only models keep message content as plain strings, so
``Chat.vision_inputs()`` has to accept strings rather than raise on them.
"""

import pytest

from chat_bricks import Chat, get_template

TEXT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Find a red shirt."},
    {"role": "assistant", "content": "search[red shirt]"},
    {"role": "user", "content": "Results: ..."},
]


def test_hf_text_model_keeps_string_content_and_returns_no_vision_inputs():
    chat = Chat(template=TEXT_MODEL, messages=MESSAGES)

    assert chat.use_list_content is False
    assert all(isinstance(m["content"], str) for m in chat.messages)
    assert dict(chat.vision_inputs()) == {}


def test_hf_vision_model_wraps_string_content_and_returns_no_vision_inputs():
    chat = Chat(template=VISION_MODEL, messages=MESSAGES)

    assert chat.use_list_content is True
    assert all(isinstance(m["content"], list) for m in chat.messages)
    assert dict(chat.vision_inputs()) == {}


@pytest.mark.parametrize("template", ["qwen2.5", TEXT_MODEL])
def test_get_vision_inputs_accepts_string_and_none_content(template):
    messages = [
        {"role": "user", "content": "plain string"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": [{"type": "text", "text": "typed part"}]},
    ]

    assert dict(get_template(template).get_vision_inputs(messages)) == {}


def test_get_vision_inputs_rejects_malformed_content():
    with pytest.raises(ValueError, match="string or a list of dicts"):
        get_template("qwen2.5").get_vision_inputs(
            [{"role": "user", "content": {"text": "not a list"}}]
        )


@pytest.mark.parametrize(
    "template, expected",
    [
        (TEXT_MODEL, False),
        (VISION_MODEL, True),
        ("qwen2.5", False),
        ("qwen2.5-vl", True),
    ],
)
def test_supports_vision(template, expected):
    assert get_template(template).supports_vision() is expected
