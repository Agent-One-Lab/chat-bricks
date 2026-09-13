"""Chat.tokenize tokenizer resolution: argument -> Chat(tokenizer=...) ->
the template's own tokenizer (HF templates) -> ValueError."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from chat_bricks import Chat, tokenize_conversations

HF_TEMPLATE = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_TEMPLATE = "qwen2.5"
MSGS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]


def test_hf_template_falls_back_to_its_own_tokenizer():
    # No tokenizer anywhere: the HFTemplate loaded one for itself, use it.
    out = Chat(template=HF_TEMPLATE, messages=MSGS).tokenize()
    assert out["input_ids"].shape[1] > 0
    # And the batch entry point works with tokenizer=None for HF names.
    batch = tokenize_conversations([MSGS], tokenizer=None, template=HF_TEMPLATE, max_length=None)
    assert batch["input_ids"].shape[1] > 0


def test_hf_fallback_matches_explicit_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_TEMPLATE)
    explicit = tokenize_conversations([MSGS], tokenizer=tok, template=HF_TEMPLATE, max_length=None)
    fallback = tokenize_conversations([MSGS], tokenizer=None, template=HF_TEMPLATE, max_length=None)
    assert explicit["input_ids"].tolist() == fallback["input_ids"].tolist()
    assert explicit["action_mask"].tolist() == fallback["action_mask"].tolist()


def test_base_template_still_raises_without_tokenizer():
    # Base templates carry no tokenizer of their own: unchanged behavior.
    with pytest.raises(ValueError, match="Tokenizer is not set"):
        Chat(template=BASE_TEMPLATE, messages=MSGS).tokenize()
