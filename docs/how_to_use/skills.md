# Skills

A **skill** is a lightweight catalogue entry — a `name` and a `description` — advertised to the model in the system prompt, typically alongside a `load_skill` tool. The model decides which skill is relevant and calls `load_skill` to pull in the full instructions on demand. This pattern keeps the base system prompt small while making a large library of capabilities discoverable.

Chat Bricks treats skills as a first-class block of the chat template, parallel to tools. Most templates don't have this concept; the ones that do (e.g. `qwen-skills`) advertise skills via a dedicated `skills_template` slot.

## When to use skills vs. tools

| | **Tools** | **Skills** |
| --- | --- | --- |
| Defined by | A JSON schema with parameters | Just a `name` + `description` |
| What the model does | Calls the tool directly | Calls `load_skill(name=...)` first, then operates on the loaded instructions |
| Typical count | A handful in scope at once | Many — only metadata is shown to the model |
| System-prompt cost | One JSON blob per tool | One line per skill |

Skills compose **with** tools, not instead of them — `load_skill` is itself a tool.

## Quickstart with `qwen-skills`

The built-in `qwen-skills` template includes the section already. Pass `skills=` and `tools=`:

```python
from chat_bricks import Chat

skills = [
    {"name": "add-numbers", "description": "Adds two integers."},
    {"name": "word-count",  "description": "Counts words in text."},
]
tools = [{
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": "Load a skill by name",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
}]

chat = Chat(template="qwen-skills", messages=[
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "hi"},
], tools=tools, skills=skills)

print(chat.prompt())
```

Renders (excerpt):

```
# Skills

You may also load one of the following skills via the load_skill tool. ...

<skills>
- add-numbers: Adds two integers.
- word-count: Counts words in text.
</skills>

You are an agent.<|im_end|>
```

## How skills render

Three pieces of the template decide what the skill block looks like:

1. **`{skills}` placeholder in `system_template`** — where the block lives.
2. **`skills_template`** — wraps the joined list, e.g. `"# Skills\n<skills>\n{skills}\n</skills>"`.
3. **`single_skill_template`** (on `SkillPolicy`) — wraps one entry; defaults to `"- {name}: {description}"`.

If a template has no `skills_template`, passing `skills=` is a silent no-op — safe to pass to any template.

## Custom skill format

Use `SkillPolicy` to change the per-entry format or the joiner:

```python
from chat_bricks import Template
from chat_bricks.policies import SkillPolicy

policy = SkillPolicy(
    single_skill_template="* {name} :: {description}",
    joiner="\n",
)

template = Template(
    name="my-skills",
    system_template="<|im_start|>system\n{system_message}{skills}<|im_end|>\n",
    skills_template="\n\n# Skills\n<skills>\n{skills}\n</skills>",
    skill_policy=policy,
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    stop_words=["<|im_end|>"],
)
```

### Truncating or rewriting descriptions

Use a `content_processor` to transform each skill before formatting — useful when descriptions are long:

```python
def truncate(skill, limit=80):
    desc = skill.get("description", "")
    if len(desc) > limit:
        return {**skill, "description": desc[:limit - 1] + "…"}
    return skill

policy_with_processor = SkillPolicy(content_processor=truncate)
```

## Skill entries can be dicts or objects

Anything with `.name` and `.description` works:

```python
from dataclasses import dataclass

@dataclass
class Skill:
    name: str
    description: str

skills = [
    {"name": "add-numbers", "description": "Adds two integers."},
    Skill("word-count", "Counts words in text."),
]
```

Missing `name` raises `TypeError`; missing `description` defaults to `""`.

## Skills in training data

Skills go through the same render pipeline as everything else, so the per-token `labels` and `action_mask` returned by `chat.tokenize(...)` cover skill-augmented prompts correctly — the skills block is part of the system message and masked out of the loss.

```python
inputs = chat.tokenize(tokenizer)
# Same input_ids / labels / action_mask shape as any other Chat
```

## Jinja parity

The same template, exported via `template.jinja_template()`, threads `skills=` through `tokenizer.apply_chat_template(messages, skills=...)` and produces identical output. See [Verification & correctness](verification.md) for how to check parity.

## Where to go next

- **[Tools and tool-call variants](tools.md)** — the `load_skill` companion side.
- **[Advanced Features](advanced_features.md)** — full `SkillPolicy` and template reference.
