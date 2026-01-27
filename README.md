# 🧩 Chat Bricks
<a href="https://chat-bricks.readthedocs.io/" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/DOC-ChatBricks-%23ffc8dd?style=for-the-badge&logo=readthedocs"></a>

*Jinja Template is Not You Need!*

Chat Bricks is a powerful and flexible template system inspired by building block toys, designed to support various LLM and VLM chat templates for training and inference.

## Key Features

- **Training and Inference**: Chat template formatted prompts, with tokenized inputs and masks.
- **Modular design**: Templates are built from configurable components.
- **Multi-modal support**: Built-in vision-language templates.
- **Jinja template generation**: Automatic HuggingFace-compatible template generation.
- **HuggingFace Integration**: Directly supports using an HF repo id as template.
- **Advanced configuration**: Fine-grained control over template behavior.

## Installation

```bash
pip install chat-bricks
```


## Quick Start

### Basic Usage

Create a chat object with a built-in template and render the prompt:

```python
from chat_bricks import Chat

# Create a chat object with template and messages
chat = Chat(
    template="qwen3",
    messages=[
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."}
    ],
)

# Render the final prompt
prompt = chat.prompt()
print(prompt)
```

### Tokenization for Training/Inference

You can easily tokenize messages for model input:

```python
from transformers import AutoTokenizer
from chat_bricks import Chat

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
chat = Chat(template="qwen2.5", messages=[{"role": "user", "content": "Hello!"}])

inputs = chat.tokenize(
    tokenizer,
    add_generation_prompt=True,  # keep generation token for inference
)

print(inputs["input_ids"])
```

### Custom Templates

Define your own template format using the `Template` class:

```python
from chat_bricks import Chat, Template

custom = Template(
    name="my-template",
    system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
    system_message="You are a concise assistant.",
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    stop_words=["<|im_end|>"],
)

chat = Chat(template=custom, messages=[{"role": "user", "content": "Hi!"}])
print(chat.prompt())
```

### Using HuggingFace Repo ID as Template

You can directly use any HuggingFace model repository ID as a template. Chat Bricks will automatically load the tokenizer's chat template:

```python
from chat_bricks import Chat

# Use a HuggingFace repo id directly
chat = Chat(
    template="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."}
    ],
)

# Render the prompt using the model's native chat template
prompt = chat.prompt()
print(prompt)
prompt_with_mask = chat.prompt_with_mask()
print(prompt_with_mask)

# Tokenize with proper masking for training
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
inputs = chat.tokenize(tokenizer, add_generation_prompt=True)
```

This feature automatically detects if the repo ID is not a built-in template and creates an `HFTemplate` that uses the tokenizer's chat template. It supports tools, generation prompts, and proper masking for training. See the [HuggingFace Templates Guide](docs/how_to_use/huggingface_templates.md) for more details.

## Documentation

For full documentation, please visit our [docs](docs/index.md) (or run `mkdocs serve` locally).

## Community

| WeChat | Discord |
| :---: | :---: |
| <img src="https://agent-one-lab.github.io/assets/agentfly/wechat.jpg" width="200" /> <br> Scan to join wechat group | <img src="https://agent-one-lab.github.io/assets/agentfly/discord.png" width="200" /> <br> [Join our discord channel](https://discord.gg/Ze5Z9QhhJ3) |
