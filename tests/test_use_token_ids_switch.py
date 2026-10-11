"""``use_token_ids`` switches ``token_ids`` splicing on and off.

With the default (True) an assistant message's ``token_ids`` are spliced verbatim into
tokenization. With False they are dropped during message normalization, so tokenization
re-encodes every turn from text exactly as if the messages carried no ids.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from chat_bricks import Chat, tokenize_conversations

HF_TEMPLATE = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_TEMPLATE = "qwen2.5"
IM_END = 151645
DRIFT_THINK = [27, 26865, 29]  # what the model samples for "<think>"; canonical re-encoding differs


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(HF_TEMPLATE)


def _messages(tokenizer, with_ids):
    """A two-turn conversation whose assistant turns were sampled with a drifted ``<think>``."""
    msgs = [{"role": "user", "content": "start"}]
    for i in range(2):
        content = f"<think>step {i}</think><action>go</action>"
        turn = {"role": "assistant", "content": content}
        if with_ids:
            turn["token_ids"] = (
                DRIFT_THINK
                + tokenizer.encode(f"step {i}</think><action>go</action>", add_special_tokens=False)
                + [IM_END]
            )
        msgs += [turn, {"role": "user", "content": f"obs {i}"}]
    return msgs


def _tok(messages, template, tokenizer, **kwargs):
    out = tokenize_conversations(
        [messages], tokenizer=tokenizer, template=template, max_length=None, **kwargs
    )
    return {k: out[k][0].tolist() for k in ("input_ids", "attention_mask", "labels", "action_mask")}


def _contains(hay, needle):
    return any(hay[i : i + len(needle)] == needle for i in range(len(hay) - len(needle) + 1))


@pytest.mark.parametrize("template", [HF_TEMPLATE, BASE_TEMPLATE])
def test_disabled_matches_the_text_path(template, tokenizer):
    if tokenizer.encode("<think>", add_special_tokens=False) == DRIFT_THINK:
        pytest.skip("tokenizer does not drift on <think>")
    text = _tok(_messages(tokenizer, with_ids=False), template, tokenizer)
    disabled = _tok(_messages(tokenizer, with_ids=True), template, tokenizer, use_token_ids=False)
    assert disabled == text
    assert not _contains(disabled["input_ids"], DRIFT_THINK)


@pytest.mark.parametrize("template", [HF_TEMPLATE, BASE_TEMPLATE])
def test_enabled_by_default_splices_sampled_ids(template, tokenizer):
    if tokenizer.encode("<think>", add_special_tokens=False) == DRIFT_THINK:
        pytest.skip("tokenizer does not drift on <think>")
    spliced = _tok(_messages(tokenizer, with_ids=True), template, tokenizer)
    assert _contains(spliced["input_ids"], DRIFT_THINK)
    assert spliced == _tok(_messages(tokenizer, with_ids=True), template, tokenizer, use_token_ids=True)


def test_chat_drops_token_ids_only_when_disabled(tokenizer):
    msgs = _messages(tokenizer, with_ids=True)
    kept = Chat(HF_TEMPLATE, msgs, tokenizer=tokenizer)
    dropped = Chat(HF_TEMPLATE, msgs, tokenizer=tokenizer, use_token_ids=False)
    assert all("token_ids" in m for m in kept.messages if m["role"] == "assistant")
    assert not any("token_ids" in m for m in dropped.messages)
    # The caller's messages are left untouched.
    assert all("token_ids" in m for m in msgs if m["role"] == "assistant")
