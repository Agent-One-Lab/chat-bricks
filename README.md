# 🧩 Chat Bricks
<a href="https://chat-bricks.readthedocs.io/" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/DOC-ChatBricks-%23ffc8dd?style=for-the-badge&logo=readthedocs"></a>

**Compose chat templates from typed bricks. Train with `labels` and `action_mask` you can trust.**

Chat Bricks is a chat-template toolkit for LLM/VLM training and inference, built on two ideas:

1. **A template is a composition** of small, typed parts — system/user/assistant blocks, section templates (`{tools}`, `{skills}`), policies, formatters, content processors, joiners. Swap any of them without rewriting Jinja.
2. **A template should be verifiable** — rendering is checked byte-for-byte against the model's official `apply_chat_template` output, and `chat.tokenize(...)` returns per-token `labels` and `action_mask` ready to drop into an SFT or RL loss.

## A quick taste

Define a template by composing bricks:

```python
from chat_bricks import (
    Chat, Template, ToolPolicy, ToolPlacement, JsonIndentedFormatter,
)

template = Template(
    name="my-agent",
    system_template="<|im_start|>system\n{system_message}{tools}<|im_end|>\n",
    system_message="You are a careful agent.",
    tools_template="\n\n# Tools\n{tools}",
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    tool_policy=ToolPolicy(
        placement=ToolPlacement.SYSTEM,
        formatter=JsonIndentedFormatter(indent=2, joiner="\n\n"),
    ),
    stop_words=["<|im_end|>"],
)

tools = [{"type": "function", "function": {
    "name": "multiply",
    "description": "Multiply two numbers",
    "parameters": {
        "type": "object",
        "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
        "required": ["x", "y"],
    },
}}]

chat = Chat(template=template,
            messages=[{"role": "user", "content": "What's 3 times 5?"}],
            tools=tools)
print(chat.prompt())
```

Renders:

```
<|im_start|>system
You are a careful agent.

# Tools
{
  "type": "function",
  "function": {
    "name": "multiply",
    "description": "Multiply two numbers",
    "parameters": {
      "type": "object",
      "properties": { "x": {"type": "number"}, "y": {"type": "number"} },
      "required": ["x", "y"]
    }
  }
}<|im_end|>
<|im_start|>user
What's 3 times 5?<|im_end|>
```

Every visible piece of that output — section ordering, the tool-block wrapper, the JSON indent, the role markers — came from a brick you can substitute. Want minified tools instead? Swap the formatter. Want tools after the user turn? Change the placement. Want a different role layout? Change `system_template` / `user_template` / `assistant_template`. Nothing rewrites the template engine.

## Two ways to define a template

**Compose your own** — typed bricks, as above. Bring your conventions, mix and match.

**Or use any HuggingFace model directly**:

```python
from chat_bricks import Chat

chat = Chat(template="Qwen/Qwen2.5-3B-Instruct", messages=[...])
# Falls back to the model's tokenizer.chat_template; masking is reconstructed
# from incremental renders so you still get correct labels + action_mask.
```

Both paths share the same `Chat` API, the same tokenizer integration, and the same correctness guarantees.

## Verified rendering + ready-to-train tensors

```python
from transformers import AutoTokenizer
from chat_bricks import Chat

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
chat = Chat(template="Qwen/Qwen2.5-3B-Instruct", messages=[
    {"role": "user", "content": "What's 3 times 5?"},
    {"role": "assistant", "content": "15."},
    {"role": "user", "content": "Now plus 2?"},
    {"role": "assistant", "content": "17."},
])

inputs = chat.tokenize(tok)
# inputs["input_ids"]      — token IDs
# inputs["labels"]         — -100 except assistant turns; drop into SFT loss
# inputs["action_mask"]    — 1 on assistant tokens, 0 elsewhere
# inputs["attention_mask"] — standard
```

The mask isn't a string-offset hack — it's reconstructed by aligning incremental renders to token spans, with model-specific overrides for templates that aren't append-only (e.g. Qwen3 drops previous thinking blocks). For the conversation above, `action_mask` flags exactly the tokens that compose `"15."` and `"17."` — nothing more.

Want to **see** the mask? Use `chat.prompt_with_mask()` to print the prompt with assistant spans color-highlighted in the terminal.

## What you get

**Composable template architecture**

- Typed bricks: `Template`, `ToolPolicy`, `SystemPolicy`, `SkillPolicy`, `GlobalPolicy`.
- Pluggable `ToolFormatter` (Qwen-style, JSON variants, YAML, custom) — swap conventions without touching Jinja.
- Two-pass section system: `{tools}` / `{skills}` placeholders, wrapper templates, per-item templates with joiners. Add a new section type in a few lines.
- Content processors for per-section transforms (truncate descriptions, filter tools by category, inject env metadata, Llama-3.2-style date stamping).
- Export to Jinja via `template.jinja_template()` for HF `tokenizer.chat_template` compatibility.

**Verifiable training-time correctness**

- Per-token `labels` and `action_mask` across multi-turn, tool-call, and skill turns.
- Byte-identical rendering vs. the official template, checked via `compare_hf_template(...)` and CI on every push.
- `Chat(template="org/model")` works with any HuggingFace repo; correctness escape hatches (`Qwen3Renderer`-style overrides) for non-append-only families.
- VLM support: vision-language templates and a registerable vision processor.

## Installation

```bash
pip install chat-bricks
```

## More examples

### Same base model, different tool conventions

Pick a built-in variant for the convention you want — no Jinja rewrites:

```python
from chat_bricks import Chat

# Tools rendered into the system prompt (Qwen's default)
Chat(template="qwen2.5", messages=..., tools=tools)

# Tools not advertised in the system prompt (describe them yourself)
Chat(template="qwen2.5-no-system-tool", messages=..., tools=tools)

```

Or roll your own with `ToolPolicy` + `ToolFormatter` — see [docs/how_to_use/tools.md](docs/how_to_use/tools.md).

### A custom tool formatter, end-to-end

```python
from chat_bricks import ToolFormatter

class XmlToolFormatter(ToolFormatter):
    def format(self, tools):
        out = []
        for t in tools:
            fn = t["function"] if "function" in t else t
            out.append(f'<tool name="{fn["name"]}">{fn.get("description","")}</tool>')
        return "\n".join(out)

    def jinja(self):  # so the same template exports cleanly to HF
        return (
            "{%- for t in tools -%}"
            '<tool name="{{ (t.function if t.function is defined else t).name }}">'
            "{{ (t.function if t.function is defined else t).description }}"
            "</tool>{%- if not loop.last %}\n{% endif %}"
            "{%- endfor -%}"
        )
```

Drop it into any template via `ToolPolicy(formatter=XmlToolFormatter())`.

### Skills + tools in the same template

The built-in `qwen-skills` template advertises a skills catalogue alongside tools:

```python
chat = Chat(
    template="qwen-skills",
    messages=[{"role": "user", "content": "Help me count words."}],
    tools=[{"type": "function", "function": {"name": "load_skill", ...}}],
    skills=[
        {"name": "add-numbers", "description": "Adds two integers."},
        {"name": "word-count",  "description": "Counts words in text."},
    ],
)
```

The skills block lives at `{skills}` in `system_template`, wrapped by `skills_template`, with each entry formatted by `SkillPolicy.single_skill_template`. See [docs/how_to_use/skills.md](docs/how_to_use/skills.md).

### Train on the last assistant turn only

```python
inputs = chat.tokenize(tok, train_on_last_turn_only=True)
# Only the final assistant turn contributes to the loss.
# Useful for RL rollouts or when earlier turns are demonstrations.
```

### Verify a template before training

```python
from chat_bricks.utils import compare_hf_template

is_equal, *_ = compare_hf_template(
    tok, "qwen2.5",
    messages=[...], tools=[...], add_generation_prompt=True,
)
assert is_equal, "Built-in render diverges from the model's official template"
```

`compare_hf_template` also checks that the *exported Jinja* round-trips to the same string — so a template you defined in Python will produce identical output when handed to any HF inference server. See [docs/how_to_use/verification.md](docs/how_to_use/verification.md).

## Documentation

Full docs at [docs/index.md](docs/index.md), or run `mkdocs serve` locally.

Recommended starting points:

- **[Use any HuggingFace model](docs/how_to_use/huggingface_templates.md)** — the HF-fallback path.
- **[Tools and tool-call variants](docs/how_to_use/tools.md)** — policies, formatters, placement, custom formats.
- **[Skills](docs/how_to_use/skills.md)** — the skills section and `SkillPolicy`.
- **[Verification & correctness](docs/how_to_use/verification.md)** — prove your template is right before you train on it.
- **[Custom Templates](docs/how_to_use/custom_templates.md)** — full reference for composing a template from scratch.

## Community

| WeChat | Discord |
| :---: | :---: |
| <img src="https://agent-one-lab.github.io/assets/agentfly/wechat.jpg" width="200" /> <br> Scan to join wechat group | <img src="https://agent-one-lab.github.io/assets/agentfly/discord.png" width="200" /> <br> [Join our discord channel](https://discord.gg/Ze5Z9QhhJ3) |
