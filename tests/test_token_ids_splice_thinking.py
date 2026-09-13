"""Splicing sampled ``token_ids`` under a *thinking* chat template (Qwen3.5).

The invariant pinned here is token-prefix continuity: for every assistant turn,
``render(messages[:i+1])`` must equal ``render(messages[:i], add_generation_prompt=True)``
followed by the sampled ids and only the template's post-terminator glue. The old
sentinel probe inserted the empty-think closer ``"\\n</think>\\n\\n"`` between the
generation prompt and the ids for Qwen3-style templates, so every spliced turn
carried four tokens the model never sampled and every sampled token was scored in a
shifted context. Both thinking modes are covered, and so is a *user* message
appended after a spliced turn (Qwen3.5 then drops earlier reasoning from its text
render, which used to make the HF prefix-diff re-emit part of the earlier turn).
"""
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from chat_bricks import tokenize_conversations

QWEN35 = "Qwen/Qwen3.5-9B"


@pytest.fixture(scope="module")
def tok():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return transformers.AutoTokenizer.from_pretrained(QWEN35, trust_remote_code=True)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{QWEN35} tokenizer unavailable offline: {e}")


def _ids(tok, text):
    return tok.encode(text, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]


def _tok(messages, tok, **kw):
    out = tokenize_conversations(
        [messages], tokenizer=tok, template=QWEN35, max_length=None,
        return_reward_mask=False, add_generation_prompt=kw.pop("add_generation_prompt", False), **kw,
    )
    return {k: out[k][0].tolist() for k in ("input_ids", "action_mask")}


def _prompt_ids(messages, tok, **kw):
    return _tok(messages, tok, add_generation_prompt=True, **kw)["input_ids"]


def _conversation(tok, thinking: bool, follow_role: str = "tool"):
    """user -> assistant(spliced) -> {tool|user} -> assistant(spliced) -> tool."""
    if thinking:
        s1, s2 = "THOUGHT_ONE\n</think>\n\nI will list files.", "THOUGHT_TWO\n</think>\n\nNow cat."
    else:
        s1, s2 = "I will list files.", "Now cat."
    follow = (
        {"role": "tool", "content": "file_a"}
        if follow_role == "tool"
        else {"role": "user", "content": "Tool call error: no tool calls found."}
    )
    return [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": s1, "token_ids": _ids(tok, s1)},
        follow,
        {"role": "assistant", "content": s2, "token_ids": _ids(tok, s2)},
        {"role": "tool", "content": "hello"},
    ]


def _trained_spans(row):
    spans, cur = [], []
    for t, m in zip(row["input_ids"], row["action_mask"]):
        if m:
            cur.append(t)
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return spans


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("follow_role", ["tool", "user"])
def test_prefix_continuity_every_turn(tok, thinking, follow_role):
    kw = {"enable_thinking": thinking}
    msgs = _conversation(tok, thinking, follow_role)
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        turn_prompt = _prompt_ids(msgs[:i], tok, **kw)
        expected = turn_prompt + m["token_ids"]
        through_turn = _tok(msgs[: i + 1], tok, **kw)["input_ids"]
        assert through_turn[: len(expected)] == expected, (
            f"turn {i}: {tok.decode(through_turn[len(turn_prompt): len(turn_prompt) + 8])!r} "
            f"!= {tok.decode(expected[len(turn_prompt): len(turn_prompt) + 8])!r}"
        )
        # After the ids only the post-terminator glue follows (a single newline here).
        assert tok.decode(through_turn[len(expected):]) == "\n"
        # And the generation prompt ends the way sampling started.
        tail = tok.decode(turn_prompt[-8:])
        assert tail.endswith("<think>\n") if thinking else tail.endswith("<think>\n\n</think>\n\n")


@pytest.mark.parametrize("thinking", [True, False])
def test_trained_span_is_exactly_the_sampled_ids(tok, thinking):
    msgs = _conversation(tok, thinking)
    row = _tok(msgs, tok, enable_thinking=thinking)
    spans = _trained_spans(row)
    sampled = [m["token_ids"] for m in msgs if m.get("role") == "assistant"]
    assert len(spans) == len(sampled)
    for span, ids in zip(spans, sampled):
        # ids verbatim, plus the post-terminator newline the template appends to the turn
        assert span[: len(ids)] == ids
        assert tok.decode(span[len(ids):]) == "\n"


def test_spliced_render_matches_template_text_render(tok):
    # With canonical ids the spliced row must equal what the HF template itself
    # renders for the same conversation (reasoning kept inside the tool loop).
    msgs = _conversation(tok, True)
    spliced = _tok(msgs, tok, enable_thinking=True)["input_ids"]
    hf = [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "I will list files.", "reasoning_content": "THOUGHT_ONE"},
        {"role": "tool", "content": "file_a"},
        {"role": "assistant", "content": "Now cat.", "reasoning_content": "THOUGHT_TWO"},
        {"role": "tool", "content": "hello"},
    ]
    text = tok.apply_chat_template(hf, tokenize=False, enable_thinking=True)
    assert tok.decode(spliced) == text


def test_user_message_after_spliced_turn_is_not_duplicated(tok):
    msgs = _conversation(tok, True, follow_role="user")
    row = _tok(msgs, tok, enable_thinking=True)
    text = tok.decode(row["input_ids"])
    assert text.count("I will list files.") == 1
    assert text.count("Tool call error") == 1
    # The earlier turn keeps the reasoning the model actually produced (spliced ids).
    assert "THOUGHT_ONE" in text
