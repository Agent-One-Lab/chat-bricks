# Train on Sampled Token IDs (RL Token Drift)

## Overview

In RL training (GRPO, PPO, GiGPO, …) the policy samples a response as **token ids**,
the framework decodes them to text to run tools or step an environment, and later
re-tokenizes the whole conversation to build the training batch. Re-encoding decoded
text is **not** the identity: the sampled ids and the canonical re-encoding can differ,
so the model is updated on tokens it never actually generated. This is
*retokenization drift* ("token drift"), and it makes every update slightly
off-policy — the importance ratios and the KL to the reference policy are computed
against a sequence that was never sampled. In practice it shows up as a KL that
keeps climbing and eventually destabilizes training.

Two common sources:

- **Non-unique tokenizations.** With Qwen2.5, the model samples `<think>` as
  `["<", "think", ">"]` = `[27, 26865, 29]`, but `tokenizer.encode("<think>")` gives
  `["<th", "ink", ">"]` = `[13708, 766, 29]`. Every turn that starts with `<think>` is
  off-policy from its first token. Measured on Qwen2.5-1.5B-Instruct, 100% of sampled
  sequences and ~4% of tokens drift.
- **Native tool calls.** Structured `tool_calls` are re-serialized by the chat template
  (`json.dumps` spacing, key order), which rarely reproduces the JSON the model actually
  emitted.

Chat Bricks fixes this at the source: an assistant message may carry the ids it was
sampled with, and tokenization **splices those ids verbatim** instead of re-encoding
the text.

## Usage

Put the generated ids on the assistant message under the `token_ids` key. Everything
else — `tokenize_conversations`, `Chat.tokenize`, masks, labels, the reward mask —
works exactly as before.

```python
from chat_bricks import tokenize_conversations

# ids exactly as vLLM returned them for this turn (eos included when the model stopped on it)
sampled_ids = [27, 26865, 29, 87, 522, 26865, 29, 151645]   # "<think>x</think>" + <|im_end|>

messages = [
    {"role": "user", "content": "You are in a room. What do you do?"},
    {
        "role": "assistant",
        "content": "<think>x</think>",          # the decoded text, used for tools / the env
        "token_ids": sampled_ids,               # the sampled ids, used for training
    },
]

inputs = tokenize_conversations(
    [messages], tokenizer=tok, template="Qwen/Qwen2.5-1.5B-Instruct", max_length=None,
    return_reward_mask=True,
)
# inputs["input_ids"] now contains [27, 26865, 29, ...] for the assistant span — the
# tokens that were actually sampled — not the re-encoded [13708, 766, 29, ...].
```

`token_ids` is optional and per message. Assistant turns without it, and every
non-assistant turn, are tokenized from text as usual, so existing pipelines are
unchanged until you opt in.

## What the tokenizer does with the ids

The generated ids cover the model's output **including the turn terminator** (e.g.
`<|im_end|>`), but not the template's structural glue around it. Chat Bricks assembles
the assistant span as:

```
[generation prompt the model was given] + token_ids + [template glue after the terminator]
```

- **Verbatim.** The ids are never decoded and never compared against the rendered text.
  A re-serialized tool call or a non-canonical `<think>` cannot cause drift or a
  fallback — what was sampled is what is trained on.
- **Structural glue, discovered per template.** The text a template places around the
  assistant content (for example the base Qwen template fuses a role-separator `\n`
  *before* the content; the HuggingFace Qwen template appends `\n` *after*
  `<|im_end|>`) is found once by a sentinel probe of the template itself and cached.
  Nothing template-specific is hardcoded, so it works for any built-in or HuggingFace
  template.
- **The terminator is matched by id.** If the ids already end with the terminator the
  glue's copy is dropped (no doubled eos); if the generation was cut off at
  `max_tokens` (no eos), the terminator is appended so the turn is still closed.
- **Masks are unchanged.** `action_mask`, `labels`, and the reward-mask position are the
  same as the text path produces; only the *source* of the assistant-span ids changes.
  With canonical ids the output is byte-identical to tokenizing the text.

## Rules for the ids you pass

- `token_ids` must be exactly the ids sampled for **that** assistant turn, in order,
  including the eos if the model stopped on it — vLLM's `CompletionOutput.token_ids`
  (or verl's `responses`, padding stripped) is the right source. Do not truncate or
  edit them.
- Pass the **full** generation even if you truncated the text for tool parsing (for
  example cutting `content` at the first `</action>`). Training on the full sampled
  sequence is what keeps the update on-policy; the truncated text is only what the
  environment saw.
- Templates that **rewrite** assistant content cannot splice it. Qwen3 strips `<think>`
  from earlier turns, so its renderer drops `token_ids` on those turns and tokenizes the
  rewritten text; only the last (unstripped) turn is spliced.
- The vision-processor path ignores `token_ids` for now.

## Verifying the splice

The drift is easy to see: tokenize once with the ids and once without, and compare the
assistant span.

```python
with_ids    = tokenize_conversations([messages], tokenizer=tok, template=tmpl, max_length=None)
plain       = [{k: v for k, v in m.items() if k != "token_ids"} for m in messages]
without_ids = tokenize_conversations([plain],    tokenizer=tok, template=tmpl, max_length=None)

span = lambda out: out["input_ids"][0][out["action_mask"][0] == 1].tolist()
print(span(with_ids) == span(without_ids))   # False whenever the sampled ids drift
```

In an RL loop this is a useful per-batch diagnostic: the fraction of rows whose spliced
span differs from the text re-encoding is the drift you were training on before.

## Using it from an RL framework

The framework's job is just to carry the ids from generation to tokenization:

1. Keep the generated ids next to the decoded text (vLLM: `output.token_ids`; verl:
   `batch["responses"]` with padding stripped via the response attention mask).
2. Set `message["token_ids"] = ids` on the assistant message you append to the
   conversation.
3. Tokenize with `tokenize_conversations` / `Chat.tokenize` as usual.

The ids survive `Chat`'s message normalization, so they can ride inside the message
dicts through any intermediate storage. This mirrors what verl-agent does natively by
training on `batch["responses"]`; with Chat Bricks the same guarantee holds for
multi-turn conversations, custom templates, and native tool-call turns.
