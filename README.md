# 🧩 Chat Bricks

**Jinja Template is Not What You Need!**

Chat Bricks is a powerful and flexible template system inspired by building block toys, designed to support various LLM and VLM chat templates for training and inference.

## Key Features

- **Training and Inference**: Chat template formatted prompts, with tokenized inputs and masks.
- **Modular design**: Templates are built from configurable components.
- **Multi-modal support**: Built-in vision-language templates.
- **Jinja template generation**: Automatic HuggingFace-compatible template generation.
- **Advanced configuration**: Fine-grained control over template behavior.

## Installation

```bash
pip install chat-bricks
```

## Supported Models

Chat Bricks comes with built-in support for many popular models, including:

- **Qwen**: `qwen2.5`, `qwen2.5-vl`, `qwen3`, `qwen3-vl-instruct`
- **Llama**: `llama-3.2`
- **Deepseek**: `deepseek-prover`, `deepseek-r1-distill-qwen`
- **GLM**: `glm-4`
- **Phi**: `phi-4`
- **Nemotron**: `nemotron`
- **Kimi**: `kimi-k2-instruct`

And more!

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
prompt = chat.render()
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
print(chat.render())
```

## Documentation

For full documentation, please visit our [docs](docs/index.md) (or run `mkdocs serve` locally).

## Community

| WeChat | Discord |
| :---: | :---: |
| <img src="https://agent-one-lab.github.io/assets/agentfly/wechat.jpg" width="200" /> <br> Scan to join wechat group | <img src="https://agent-one-lab.github.io/assets/agentfly/discord.png" width="200" /> <br> [Join our discord channel](https://discord.gg/Ze5Z9QhhJ3) |

