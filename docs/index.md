# 🧩 Chat Bricks

*Jinja Template is Not You Need!*

Chat Bricks is a powerful and flexible template system inspired by building block toys, designed to support various LLM and VLM chat templates for training and inference.

## Key Features

- **Training and Inference**: Chat template formatted prompts, with tokenized inputs and masks.
- **Modular design**: Templates are built from configurable components.
- **Multi-modal support**: Vision-language templates are built in.
- **Jinja template generation**: Automatic HuggingFace-compatible template generation.
- **HuggingFace Integration**: Directly supports using an HF repo id as template.
- **Advanced configuration**: Fine-grained control over template behavior.

## Quickstart

```python
from chat_bricks import get_template, Chat

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


| WeChat | Discord |
| :---: | :---: |
| <img src="https://agent-one-lab.github.io/assets/agentfly/wechat.jpg" width="200" /> <br> Scan to join wechat group | <img src="https://agent-one-lab.github.io/assets/agentfly/discord.png" width="200" /> <br> [Join our discord channel](https://discord.gg/Ze5Z9QhhJ3) |

