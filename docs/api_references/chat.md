# Chat API Reference

`Chat` is the high-level interface that wraps messages and `Template` to render prompts and build tokenized inputs. This is the interface that we recommend users to use.

## Minimal Example
```python
from chat_bricks import Chat, get_template
from chat_bricks.utils.tokenize import VisionProcessor

chat = Chat(
    template="qwen3",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Render prompt text
prompt = chat.render()

# Tokenize for training or inference
inputs = chat.tokenize(
    tokenizer,
    processor=processor,          # required for vision models
    add_generation_prompt=True,   # keep generation token for inference
    tools=tools                   # optional tool definitions
)
```

::: chat_bricks.chat.Chat
