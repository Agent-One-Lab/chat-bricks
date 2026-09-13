"""Tests for per-message ``token_ids`` splicing during tokenization.

Background: in multi-turn RL we sample assistant turns with vLLM, decode them to
text to feed the environment, then re-tokenize the whole conversation for the
training update. Re-encoding decoded text is *not* identity -- e.g. Qwen2.5
samples ``<think>`` as [27, 26865, 29] but canonically re-encodes it as
[13708, 766, 29]; and native tool calls re-serialize structured ``tool_calls``
back to JSON with possibly different spacing/key order. Either way the training
ids drift from what was actually sampled -> off-policy updates.

Fix: an assistant message may carry ``token_ids`` (the exact ids vLLM generated,
eos included). Tokenization then uses those ids **verbatim** for the assistant
content span and appends only the template's structural post-terminator glue
(e.g. a trailing ``\\n``), stitched by *token id* (no text comparison, no
re-serialization). These tests pin: identity when ids are canonical; drift and
tool-call serialization preserved; correct eos handling / reward placement;
truncation (no eos) handled; and content-rewriting templates (Qwen3) safely
withholding the splice.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from chat_bricks import tokenize_conversations

HF_TEMPLATE = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_TEMPLATE = "qwen2.5"
IM_END = 151645
NL = 198  # tokenizer.encode("\n") == [198]
DRIFT_THINK = [27, 26865, 29]  # what the model samples for "<think>"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(HF_TEMPLATE)


def _tok_one(messages, template, tokenizer, reward=False):
    out = tokenize_conversations(
        [messages],
        tokenizer=tokenizer,
        template=template,
        max_length=None,
        return_reward_mask=reward,
    )
    keys = ["input_ids", "attention_mask", "labels", "action_mask"]
    if reward:
        keys.append("reward_mask")
    return {k: out[k][0].tolist() for k in keys}


def _sublist_index(hay, needle):
    for i in range(len(hay) - len(needle) + 1):
        if hay[i : i + len(needle)] == needle:
            return i
    return -1


def _sublist_count(hay, needle):
    return sum(
        1
        for i in range(len(hay) - len(needle) + 1)
        if hay[i : i + len(needle)] == needle
    )


def _gen_ids(content, tokenizer, eos=True):
    """The ids vLLM would produce for an assistant turn: content (+ eos)."""
    ids = tokenizer.encode(content, add_special_tokens=False)
    return ids + [IM_END] if eos else ids


# --------------------------------------------------------------------------- #
# T1 - identity: canonical ids splice to *exactly* the pure-text path.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("template", [HF_TEMPLATE, BASE_TEMPLATE])
def test_identity_when_ids_are_canonical(template, tokenizer):
    content = "<think>let me think</think><action>go north</action>"
    text_msgs = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": content},
    ]
    ids_msgs = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": content, "token_ids": _gen_ids(content, tokenizer)},
    ]
    text = _tok_one(text_msgs, template, tokenizer)
    spliced = _tok_one(ids_msgs, template, tokenizer)
    assert spliced["input_ids"] == text["input_ids"]
    # The trained span of a spliced turn is exactly the sampled ids (the model's
    # continuation of the generation prompt). The base ``qwen2.5`` template fuses
    # the role-separator newline into its content slot, so its text path trains
    # that newline although it belongs to the generation prompt; the spliced path
    # masks it, and the two masks differ only there.
    trained = [t for t, m in zip(spliced["input_ids"], spliced["action_mask"]) if m]
    ids = _gen_ids(content, tokenizer)
    assert trained[: len(ids)] == ids
    # Anything after the ids is the template's post-terminator glue (HF: a newline).
    assert all(t == NL for t in trained[len(ids):])
    if template == HF_TEMPLATE:
        assert spliced["action_mask"] == text["action_mask"]
        assert spliced["labels"] == text["labels"]
    else:
        diff = [i for i, (a, b) in enumerate(zip(text["action_mask"], spliced["action_mask"])) if a != b]
        assert len(diff) == 1 and text["input_ids"][diff[0]] == NL


# --------------------------------------------------------------------------- #
# T7 - regression: messages without token_ids behave exactly as before.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("template", [HF_TEMPLATE, BASE_TEMPLATE])
def test_no_token_ids_is_unchanged(template, tokenizer):
    msgs = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "<think>a</think><action>x</action>"},
    ]
    out = _tok_one(msgs, template, tokenizer)
    assert any(out["action_mask"]), "expected a trained assistant span"
    assert out["input_ids"].count(IM_END) >= 1


# --------------------------------------------------------------------------- #
# T2 - drift preserved: the sampled (drifted) ids survive verbatim into training.
# --------------------------------------------------------------------------- #
def test_drift_preserved_hf(tokenizer):
    if tokenizer.encode("<think>", add_special_tokens=False) == DRIFT_THINK:
        pytest.skip("tokenizer does not drift on <think>")
    content = "<think>x</think>"
    drifted = DRIFT_THINK + tokenizer.encode("x</think>", add_special_tokens=False) + [IM_END]
    text_msgs = [{"role": "user", "content": "s"}, {"role": "assistant", "content": content}]
    ids_msgs = [
        {"role": "user", "content": "s"},
        {"role": "assistant", "content": content, "token_ids": drifted},
    ]
    text = _tok_one(text_msgs, HF_TEMPLATE, tokenizer)
    spliced = _tok_one(ids_msgs, HF_TEMPLATE, tokenizer)
    assert _sublist_index(spliced["input_ids"], DRIFT_THINK) != -1
    assert _sublist_index(text["input_ids"], DRIFT_THINK) == -1
    assert spliced["input_ids"] != text["input_ids"]


# --------------------------------------------------------------------------- #
# T3 - eos / reward: exactly one eos in the assistant span (no doubling), and
# reward lands where the text path puts it (base -> <|im_end|>, HF -> \n).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "template,reward_tok", [(HF_TEMPLATE, NL), (BASE_TEMPLATE, IM_END)]
)
def test_no_double_eos_and_reward_position(template, reward_tok, tokenizer):
    content = "<think>x</think><action>go</action>"
    ids_msgs = [
        {"role": "user", "content": "s"},
        {"role": "assistant", "content": content, "token_ids": _gen_ids(content, tokenizer)},
    ]
    text_msgs = [{"role": "user", "content": "s"}, {"role": "assistant", "content": content}]
    spliced = _tok_one(ids_msgs, template, tokenizer, reward=True)
    text = _tok_one(text_msgs, template, tokenizer, reward=True)
    ii = spliced["input_ids"]
    assert not any(ii[i] == IM_END and ii[i + 1] == IM_END for i in range(len(ii) - 1))
    assert spliced["reward_mask"].index(1) == text["reward_mask"].index(1)
    assert ii[spliced["reward_mask"].index(1)] == reward_tok


# --------------------------------------------------------------------------- #
# T-tool - NATIVE TOOL CALLS: the sampled (compact) tool-call ids are preserved
# verbatim, even though the template re-serializes tool_calls with different
# spacing. This case used to *drift* under the old text-comparison approach.
# --------------------------------------------------------------------------- #
def test_native_tool_call_preserved_verbatim(tokenizer):
    tool_calls = [
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    ]
    # Model sampled COMPACT json (no spaces after : and ,); template re-dumps spaced.
    sampled = '<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>'
    compact_ids = tokenizer.encode(sampled, add_special_tokens=False)
    token_ids = compact_ids + [IM_END]

    ids_msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls, "token_ids": token_ids},
    ]
    text_msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
    ]
    spliced = _tok_one(ids_msgs, HF_TEMPLATE, tokenizer)
    text = _tok_one(text_msgs, HF_TEMPLATE, tokenizer)

    # The exact sampled (compact) tokenization is present verbatim...
    assert _sublist_index(spliced["input_ids"], compact_ids) != -1
    # ...and it is NOT what the text path (spaced re-serialization) produced.
    assert _sublist_index(text["input_ids"], compact_ids) == -1
    assert spliced["input_ids"] != text["input_ids"]
    # Still exactly one eos, no doubling.
    ii = spliced["input_ids"]
    assert not any(ii[i] == IM_END and ii[i + 1] == IM_END for i in range(len(ii) - 1))


# --------------------------------------------------------------------------- #
# T-trunc - truncation: token_ids with no eos (max-length cutoff) still get the
# terminator appended so the turn is properly closed.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("template", [HF_TEMPLATE, BASE_TEMPLATE])
def test_truncated_generation_gets_terminator(template, tokenizer):
    content = "<think>partial output with no end"
    no_eos = _gen_ids(content, tokenizer, eos=False)  # NO eos
    ids_msgs = [
        {"role": "user", "content": "s"},
        {"role": "assistant", "content": content, "token_ids": no_eos},
    ]
    out = _tok_one(ids_msgs, template, tokenizer)
    # The generated ids are present verbatim...
    assert _sublist_index(out["input_ids"], no_eos) != -1
    # ...and a terminator was appended right after them (turn is closed).
    j = _sublist_index(out["input_ids"], no_eos)
    assert out["input_ids"][j + len(no_eos)] == IM_END


# --------------------------------------------------------------------------- #
# T4 / T5 - multi-turn and partial coverage.
# --------------------------------------------------------------------------- #
def test_multi_turn_each_assistant_spliced(tokenizer):
    if tokenizer.encode("<think>", add_special_tokens=False) == DRIFT_THINK:
        pytest.skip("no drift on this tokenizer")
    ids1 = DRIFT_THINK + tokenizer.encode("a</think>", add_special_tokens=False) + [IM_END]
    ids2 = DRIFT_THINK + tokenizer.encode("b</think>", add_special_tokens=False) + [IM_END]
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>a</think>", "token_ids": ids1},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "<think>b</think>", "token_ids": ids2},
    ]
    out = _tok_one(msgs, HF_TEMPLATE, tokenizer)
    assert _sublist_count(out["input_ids"], DRIFT_THINK) == 2


def test_partial_token_ids(tokenizer):
    if tokenizer.encode("<think>", add_special_tokens=False) == DRIFT_THINK:
        pytest.skip("no drift on this tokenizer")
    ids1 = DRIFT_THINK + tokenizer.encode("a</think>", add_special_tokens=False) + [IM_END]
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>a</think>", "token_ids": ids1},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "<think>b</think>"},  # no token_ids
    ]
    text_all = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>a</think>"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "<think>b</think>"},
    ]
    out = _tok_one(msgs, HF_TEMPLATE, tokenizer)
    text = _tok_one(text_all, HF_TEMPLATE, tokenizer)
    assert _sublist_index(out["input_ids"], DRIFT_THINK) != -1
    assert out["input_ids"] != text["input_ids"]
    # Only the first (spliced) turn carries the drifted block.
    assert _sublist_count(out["input_ids"], DRIFT_THINK) == 1
