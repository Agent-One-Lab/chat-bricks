# Chat API Reference

`Chat` is the high-level interface that wraps messages and `Template` to render prompts and build tokenized inputs. This is the interface that we recommend users to use.

## Minimal Example
```python
from chat_bricks import Chat, get_template
from transformers import AutoTokenizer

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Retrieves current weather for the given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Units the temperature will be returned in."
            }
        },
        "required": ["location", "units"]
    }
}]

chat = Chat(
    template="qwen3",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Render prompt text
prompt = chat.prompt(tools=tools)
print(prompt)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
# Tokenize for training or inference
inputs = chat.tokenize(
    tokenizer,
    add_generation_prompt=True,   # keep generation token for inference
    tools=tools                   # optional tool definitions
)
```

::: chat_bricks.chat.Chat
