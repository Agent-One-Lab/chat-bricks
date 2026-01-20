# Using Templates and Chat

This guide shows how to import and use the core interfaces for chat templates: the high-level `Chat` helper and the lower-level `Template`.

## Imports

```python
from chat_bricks import Chat, get_template
from chat_bricks.templates import Template
```

## Use a Built-in Template with Chat

```python
# Grab a built-in template by name
chat = Chat(
    template="qwen3",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)

# Render a prompt string
prompt = chat.render()
print(prompt)
```

## Tokenize for Training/Inference

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

inputs = chat.tokenize(
    tokenizer,
    add_generation_prompt=True,  # keep generation token for inference
)

print(inputs["input_ids"][:20])
```

For vision or tools, pass `processor` and `tools` to `tokenize`:

```python
inputs = chat.tokenize(
    tokenizer=tokenizer,
    processor=processor,   # required for vision templates
    tools=tools,           # optional tool definitions
)
```

## Build a Custom Template

```python
custom = Template(
    name="my-template",
    system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
    system_message="You are a concise assistant.",
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    stop_words=["<|im_end|>"],
)

chat = Chat(template=custom, messages=[{"role": "user", "content": "Hi!"}])
print(chat.render())
```

## Handy Chat Methods

- `add_user_message(content)` / `add_assistant_message(content)` to append turns.
- `append(message_dict)` to add a raw message.
- `render(add_generation_prompt=False, **kwargs)` to get the formatted prompt.
- `tokenize(tokenizer, processor=None, add_generation_prompt=False, tools=None, **kwargs)` to produce token IDs, masks, labels, and action masks.
