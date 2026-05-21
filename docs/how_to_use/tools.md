# Tools and tool-call variants

A model's official chat template hardcodes **one** way to advertise tools, **one** way to render tool calls, and **one** way to format tool observations. If you want to train the same base model with a different tool convention — Qwen-style `<tool_call>` blocks, OpenAI-style JSON, a custom XML wrapper, observations inside or outside the user turn — the publisher template gives you nothing.

Chat Bricks separates these decisions into composable pieces so you can pick, mix, or replace them without touching Jinja.

## The four moving parts

| Piece | What it controls |
| --- | --- |
| **`ToolPlacement`** | *Where* the tool catalogue appears: in the system message, as a first/last user turn, or as its own role. |
| **`ToolFormatter`** | *How* the catalogue is serialized (JSON variants, YAML, your own). |
| **`single_tool_call_template`** | How **one** assistant tool call is rendered (joined into `tool_calls_template`). |
| **`single_observation_template`** | How **one** tool result is rendered (joined into `observations_template`). |

The first two govern what the model *sees* about available tools. The last two govern how tool *use* is rendered into the training prompt — which is also what determines correct loss masking.

## Picking a built-in variant

Several built-ins ship for the Qwen family alone, with different conventions:

```python
from chat_bricks import Chat

tools = [...]
messages = [...]

# Tools rendered into the system prompt (Qwen's default)
Chat(template="qwen2.5", messages=messages, tools=tools)

# No system-prompt tool catalogue — describe tools yourself
Chat(template="qwen2.5-no-system-tool", messages=messages, tools=tools)

# Tool-call generation tuned for toolgen-style training
Chat(template="toolgen-qwen2.5", messages=messages, tools=tools)
```

These produce different rendered prompts on the same base model. Pick the one that matches how you want training data to look.

## Swapping the formatter

`ToolPolicy` is the glue. To change how tools serialize without re-authoring the template, copy the existing template and replace its `tool_policy.formatter`:

```python
from chat_bricks import Chat, get_template, ToolPolicy, ToolPlacement
from chat_bricks import JsonIndentedFormatter

base = get_template("qwen2.5")
variant = base.copy()
variant.name = "qwen2.5-pretty-tools"
variant.tool_policy = ToolPolicy(
    placement=ToolPlacement.SYSTEM,
    formatter=JsonIndentedFormatter(indent=2, joiner="\n\n"),
)

chat = Chat(template=variant, messages=messages, tools=tools)
print(chat.prompt())
```

Available built-in formatters:

- `JsonQwenFormatter` — Qwen's default (per-tool JSON, newline-joined)
- `JsonMinifiedFormatter` — single line, no whitespace
- `JsonIndentedFormatter` — pretty-printed (used by some Mistral variants)
- `JsonCompactFormatter` — whole catalogue as one JSON array
- `JsonFormatterNoBreakLine` — no joiner between objects
- `YamlFormatter` — requires `pyyaml` extra

## Writing a custom formatter

Subclass `ToolFormatter` and implement both `format` (Python path) and `jinja` (so the same template can be exported for HF compatibility):

```python
from chat_bricks import ToolFormatter

class XmlToolFormatter(ToolFormatter):
    def format(self, tools):
        out = []
        for t in tools:
            fn = t["function"] if "function" in t else t
            out.append(
                f"<tool name=\"{fn['name']}\">{fn.get('description','')}</tool>"
            )
        return "\n".join(out)

    def jinja(self):
        return (
            "{%- for t in tools -%}"
            "<tool name=\"{{ (t.function if t.function is defined else t).name }}\">"
            "{{ (t.function if t.function is defined else t).description }}"
            "</tool>{%- if not loop.last %}\n{% endif %}"
            "{%- endfor -%}"
        )
```

Plug it in via `ToolPolicy(formatter=XmlToolFormatter())`.

## Placement

Where the catalogue appears changes how the model attends to it during training. The choices:

```python
from chat_bricks import ToolPlacement

ToolPlacement.SYSTEM       # inside the system message (most common)
ToolPlacement.FIRST_USER   # as an extra first-user turn
ToolPlacement.LAST_USER    # appended to the last user turn
ToolPlacement.SEPARATE     # its own dedicated role
```

Match this to the model's pre-training distribution if you care about not drifting too far from the model's expected layout.

## Tool call and observation rendering

When the assistant emits a tool call, two templates control the output:

- `single_tool_call_template` — formats one `{"type": "function", "function": {...}}` entry.
- `tool_calls_template` — wraps the joined sequence (e.g. `"<tool_call>\n{tool_calls}\n</tool_call>"`).

Likewise for observations: `single_observation_template` per entry, `observations_template` for the wrapper.

These templates are what the renderer's mask aligner uses to determine which tokens are loss-bearing. Picking the right format here is what gives you correct labels for tool-using assistant turns.

## Verifying your choice

Before you train, confirm the rendered prompt round-trips through the model's official template. See [Verification & correctness](verification.md).

## Where to go next

- **[Advanced Features](advanced_features.md)** — full reference for `ToolPolicy`, `ToolContentProcessor`, `SystemPolicy`, custom processors, and policy inheritance.
- **[Skills](skills.md)** — compose a skill catalogue alongside tools in the same template.
