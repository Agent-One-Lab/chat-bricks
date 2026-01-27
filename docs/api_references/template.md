# Template API Reference

`Template` defines how messages are formatted, what system messages are used, and how tools or multimodal content are inserted.

## Minimal Example

```python
from chat_bricks.templates import Template

template = Template(
    name="custom",
    system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
    system_message="You are a helpful assistant.",
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    stop_words=["<|im_end|>"],
)
```

::: chat_bricks.templates.Template

::: chat_bricks.templates.Renderer

::: chat_bricks.templates.JinjaGenerator
