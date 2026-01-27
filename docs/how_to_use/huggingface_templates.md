# Using HuggingFace Templates

## Overview

Chat Bricks supports directly using HuggingFace model repository IDs as templates. When you pass a HuggingFace repo ID (e.g., `"Qwen/Qwen2.5-3B-Instruct"`) to the `Chat` class, it automatically creates an `HFTemplate` that uses the tokenizer's built-in chat template. This allows you to work with any model that has a chat template defined in its tokenizer, without needing to manually configure the template format.

!!! warning
    Current implementation only works with chat templates that directly append elements when getting more turns of messages. It may not work correctly with templates that modify previous prompt content (e.g., Qwen3's template which deletes previous thinking content).

## Basic Usage

### Simple Chat with HF Repo ID

The simplest way to use a HuggingFace template is to pass the repo ID directly to `Chat`:

```python
from chat_bricks import Chat

# Use a HuggingFace repo ID directly
chat = Chat(
    template="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."}
    ],
)

# Render the prompt
prompt = chat.prompt()
print(prompt)
```

### How It Works

When you pass a string that's not a registered built-in template name, Chat Bricks automatically:

1. Loads the tokenizer from the HuggingFace repository
2. Checks if the tokenizer has a `chat_template` attribute
3. Creates an `HFTemplate` instance that uses the tokenizer's chat template
4. Handles message formatting, masking, and tokenization automatically

```python
from chat_bricks import Chat
from chat_bricks.templates import HFTemplate

# These are equivalent:
chat1 = Chat(template="Qwen/Qwen2.5-3B-Instruct", messages=messages)

# Or explicitly create HFTemplate
hf_template = HFTemplate("Qwen/Qwen2.5-3B-Instruct")
chat2 = Chat(template=hf_template, messages=messages)
```

## Supported Features

### 1. Basic Chat Rendering

HFTemplate supports standard chat message rendering:

```python
from chat_bricks import Chat

messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "Tell me more about it."}
]

chat = Chat(template="Qwen/Qwen2.5-3B-Instruct", messages=messages)
prompt = chat.prompt()
print(prompt)
```

### 2. System Messages

System messages are automatically handled according to the model's chat template:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

chat = Chat(template="Qwen/Qwen2.5-3B-Instruct", messages=messages)
prompt = chat.prompt()
```

### 3. Generation Prompts

Add generation prompts for inference:

```python
chat = Chat(
    template="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Add generation prompt for inference
prompt = chat.prompt(add_generation_prompt=True)
print(prompt)
```

### 4. Tool Support

HFTemplate supports tool definitions and tool calls:

```python
messages = [
    {"role": "user", "content": "What's the weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}'
                }
            }
        ]
    },
    {"role": "tool", "content": "Sunny, 22°C"}
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }
]

chat = Chat(
    template="Qwen/Qwen2.5-3B-Instruct",
    messages=messages,
    tools=tools
)

prompt = chat.prompt(tools=tools)
print(prompt)
```

### 5. Tokenization with Masking

HFTemplate provides proper tokenization with masking for training:

```python
from transformers import AutoTokenizer
from chat_bricks import Chat

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I am fine, thank you."}
]

chat = Chat(template="Qwen/Qwen2.5-3B-Instruct", messages=messages)

# Tokenize with proper masking
inputs = chat.tokenize(
    tokenizer=tokenizer,
    add_generation_prompt=False  # Set to True for inference
)

print("Input IDs:", inputs["input_ids"])
print("Attention Mask:", inputs["attention_mask"])
print("Labels:", inputs["labels"])  # -100 for non-assistant tokens
print("Action Mask:", inputs["action_mask"])  # 1 for assistant tokens
```

The tokenization automatically:
- Masks non-assistant tokens (sets labels to -100)
- Creates action masks (1 for assistant tokens, 0 for others)
- Handles BOS tokens if needed
- Maintains proper alignment with the chat template

### 6. Prompt with Mask Visualization

You can visualize which parts of the prompt are masked:

```python
chat = Chat(
    template="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
)

# Get prompt with color-coded masking
prompt_with_mask = chat.prompt_with_mask()
print(prompt_with_mask)  # Red = masked, Green = not masked
```

## Advanced Usage

### Direct HFTemplate Creation

You can also create an `HFTemplate` directly:

```python
from chat_bricks.templates import HFTemplate
from chat_bricks import Chat

# Create HFTemplate directly
hf_template = HFTemplate("Qwen/Qwen2.5-3B-Instruct")

# Use it with Chat
chat = Chat(template=hf_template, messages=messages)
prompt = chat.prompt()
```

### Accessing the Tokenizer

The `HFTemplate` stores the loaded tokenizer, which you can access:

```python
from chat_bricks.templates import HFTemplate

hf_template = HFTemplate("Qwen/Qwen2.5-3B-Instruct")
tokenizer = hf_template.tokenizer

# Use the tokenizer directly
print(tokenizer.chat_template)
```

### Getting the Jinja Template

You can retrieve the original Jinja chat template:

```python
from chat_bricks.templates import HFTemplate

hf_template = HFTemplate("Qwen/Qwen2.5-3B-Instruct")
jinja_template = hf_template.jinja_template()
print(jinja_template)
```


## Limitations and Considerations

### Template Compatibility

HFTemplate works best with chat templates that directly append elements when getting more turns of messages. It may not work correctly with templates that modify previous prompt content (e.g., Qwen3's template which deletes previous thinking content).

### Vision Models

For vision-language models, HFTemplate automatically detects if the model is a vision model and adjusts message formatting accordingly:

```python
# For vision models, messages are kept in vision format
chat = Chat(
    template="Qwen/Qwen2.5-VL-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "image": "path/to/image.jpg"}
            ]
        }
    ]
)
```

### Error Handling

If a tokenizer doesn't have a chat template, an error will be raised:

```python
try:
    hf_template = HFTemplate("some-model-without-chat-template")
except Exception as e:
    print(f"Error: {e}")
    # Error: Tokenizer from some-model-without-chat-template does not have a chat_template.
```


## Comparison with Built-in Templates

| Feature | Built-in Templates | HFTemplate |
|---------|-------------------|------------|
| Setup | Pre-configured | Automatic from HF repo |
| Customization | Full control | Limited to tokenizer's template |
| Masking | Custom logic | Automatic via iterative rendering |
| Tools | Full support | Depends on tokenizer support |
| Vision | Built-in support | Auto-detected for vision models |
| Performance | Optimized | May be slower (loads tokenizer) |

## Best Practices

1. **Use HFTemplate for**: Quick prototyping, working with new models, ensuring compatibility with HuggingFace's chat templates
2. **Use Built-in Templates for**: Production systems, custom requirements, better performance, fine-grained control
3. **Cache Tokenizers**: If using the same model multiple times, consider caching the HFTemplate instance
4. **Verify Template**: Always test that the generated prompts match your expectations
5. **Check Masking**: Use `prompt_with_mask()` to verify masking behavior for training


### Masking Issues

If masking doesn't work as expected:
- Check if the template modifies previous content (not supported)
- Verify the template format with `prompt_with_mask()`
- Consider using a built-in template if available
