# Verification & correctness

A subtly wrong chat template doesn't crash — it quietly degrades the model and you blame the data. Chat Bricks treats correctness as something to **prove**, not assume.

There are two correctness claims to verify, and you should check both before training on any non-trivial template:

1. **Rendering parity** — the prompt string Chat Bricks produces is byte-identical to the model's official `apply_chat_template` output.
2. **Tokenization parity** — `chat.tokenize(...)` produces the same `input_ids` you would get by tokenizing the official prompt with `add_special_tokens=False` (i.e. the rendered string already contains any BOS markers the template emits).

Mask alignment follows from these two: if rendering and tokenization both match, the per-token `labels` and `action_mask` Chat Bricks emits correspond to exactly the assistant spans you would expect.

## One-shot check with `compare_hf_template`

```python
from transformers import AutoTokenizer
from chat_bricks.utils import compare_hf_template

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 3 times 5?"},
    {"role": "assistant", "content": "15"},
]

(is_equal,
 is_equal_between_implemented_prompts,
 is_equal_between_jinja_prompts,
 official_prompt,
 implemented_prompt,
 implemented_jinja_prompt,
 highlighted_prompt) = compare_hf_template(
    tokenizer, "qwen2.5", messages=messages, add_generation_prompt=True,
)

assert is_equal,                            "Python render diverges from official template"
assert is_equal_between_jinja_prompts,      "Exported Jinja diverges from Python render"
assert is_equal_between_implemented_prompts, "Mask-highlighted render diverges"
```

The three booleans cover three correctness contracts:

- **`is_equal`** — Python render vs. the official `apply_chat_template` output.
- **`is_equal_between_jinja_prompts`** — Chat Bricks's *exported* Jinja template, when fed back through `tokenizer.apply_chat_template`, produces the same string as the Python render. This is what lets you ship the same template to inference servers that only consume Jinja.
- **`is_equal_between_implemented_prompts`** — the prompt with mask annotations (used internally to compute `action_mask`) round-trips to the same string when stripped.

If any of the three fail, `highlighted_prompt` shows the assistant spans colored in the terminal — use it to find where the divergence is.

## Verifying tokenization

For Llama-style templates that emit BOS markers in the rendered string (`<|begin_of_text|>`), tokenizing the result with `add_special_tokens=True` would double-prepend the BOS. The right comparison is:

```python
prompt = chat.prompt()
hf_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
implemented = chat.tokenize(tokenizer)

assert torch.equal(hf_inputs["input_ids"], implemented["input_ids"])
```

For templates that don't include special tokens in the rendered string (most Qwen variants), `add_special_tokens=True` is also safe because there's nothing to add.

## What the test suite checks

Every built-in template is checked against the model's official template in CI on every push and PR. The relevant suites:

| File | What it asserts |
| --- | --- |
| `tests/test_builtin_templates/test_*_full_align.py` | `compare_hf_template` byte-equality across a matrix of system/no-system × tools/no-tools × multi-turn × generation-prompt cases |
| `tests/test_builtin_templates/test_text_templates_tokenize.py` | `chat.tokenize` produces the same `input_ids` as the official tokenizer (with `add_special_tokens=False`) |
| `tests/test_builtin_templates/test_skills_jinja_parity.py` | Python render and exported-Jinja render match across `(tools, skills) ∈ {neither, tools-only, skills-only, both}` |

When you add a new template or change a policy, run those suites locally before merging:

```bash
pytest tests/test_builtin_templates/ -k "not llama and not kimi"
```

The `-k` filter skips gated-model tests that can't run in CI.

## When diff-based masking is *approximate*

The `HFTemplate` path (using a HuggingFace repo as the template) reconstructs masks by **diffing** incremental renders of the conversation. This works on append-only templates — i.e. each new turn adds text without modifying earlier turns.

It silently produces wrong masks on templates that **mutate prior content**. Known case: Qwen3 drops previous thinking blocks from the history when rendering a new turn, so the diff misaligns. Chat Bricks ships a hand-written `Qwen3Template`/`Qwen3Renderer` for this reason.

If you're using `Chat(template="some/repo-id", ...)` for a new family:

1. Run `compare_hf_template` on a multi-turn conversation **with assistant turns of varying length**.
2. If `is_equal` holds but the model still produces odd loss values, the template may be non-append-only. Open an issue or write a `*Renderer` subclass following the `Qwen3Renderer` pattern.

## Pre-flight checklist for new models

Before kicking off a training run with a model you haven't used before:

1. **Render check** — `compare_hf_template(tokenizer, template_name, messages, tools, add_generation_prompt=True)` returns `is_equal=True`.
2. **Tokenization check** — `chat.tokenize` matches `tokenizer(prompt, add_special_tokens=False)`.
3. **Mask sanity** — `inputs["action_mask"].sum()` matches roughly the assistant content length you'd expect (decode a few `inputs["input_ids"][mask == 1]` spans to eyeball).
4. **Multi-turn check** — repeat (1)–(3) with a 4-turn conversation including a tool call.

If any step fails, fix it before you train, not after.

## Where to go next

- **[Use any HuggingFace model](huggingface_templates.md)** — how `HFTemplate` and the diff-based mask path work.
- **[Tools and tool-call variants](tools.md)** — verify a custom tool format produces correct masks.
