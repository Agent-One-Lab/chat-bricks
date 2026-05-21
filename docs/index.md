# 🧩 Chat Bricks

*Correct, verifiable chat-template rendering and per-token loss masks for LLM/VLM training — with any HuggingFace model.*

Chat Bricks gives you the things `apply_chat_template` doesn't: **per-token `labels` and `action_mask` for multi-turn SFT and RL**, **swappable tool-call formats** for the same base model, and a **first-class `skills` block**. Rendering is verified byte-identical against the model's official template, so you can trust what hits your loss function.

## The problem

When you train on multi-turn or tool-using conversations, you need a per-token mask that says *"compute loss on these assistant tokens, ignore everything else."* HuggingFace's `apply_chat_template` doesn't produce this — `return_assistant_tokens_mask` only works on templates that ship with explicit `{% generation %}` markers, which most don't. Hand-rolling a mask from string offsets silently breaks on multi-turn, tool-call turns, or non-append-only templates. A wrong mask doesn't crash — it quietly degrades your model and you blame the data.

Chat Bricks reconstructs the mask by aligning incremental renders to token spans, with model-specific overrides for templates that aren't append-only. Rendering is checked byte-for-byte against each model's official chat template in CI.

## What you get

- **Loss masking that works.** Per-token `labels` and `action_mask` across multi-turn, tool-call, and skill turns. Byte-identical rendering verified against the official template.
- **Tool-call variant control.** Swap tool format on the same base model via `ToolPolicy` + `ToolFormatter` — no Jinja rewrites. See [Tools and tool-call variants](how_to_use/tools.md).
- **Skills as a first-class block.** Advertise `(name, description)` pairs in the system prompt via `skills_template`. See [Skills](how_to_use/skills.md).
- **Any HuggingFace model, out of the box.** `Chat(template="org/model", ...)` falls back to the tokenizer's chat template with masking reconstructed by diffing. See [Use any HuggingFace model](how_to_use/huggingface_templates.md).
- **Verified correctness.** `compare_hf_template(...)` and CI parity tests for every built-in template. See [Verification & correctness](how_to_use/verification.md).
- **VLM support.** Vision-language templates and a registerable vision processor. See [Vision Templates](how_to_use/vision_templates.md).

## 60-second SFT example

```python
from transformers import AutoTokenizer
from chat_bricks import Chat

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

chat = Chat(template="Qwen/Qwen2.5-3B-Instruct", messages=[
    {"role": "user", "content": "What is 3 times 5?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "multiply",
         "arguments": {"x": 3, "y": 5}}}]},
    {"role": "tool", "content": "15"},
    {"role": "assistant", "content": "It's 15."},
])

inputs = chat.tokenize(tokenizer)
# inputs["input_ids"], inputs["labels"], inputs["action_mask"], inputs["attention_mask"]
```

Continue with the [Quick Start](quick_start/use_template.md) or jump to any of the how-to pages above.


| WeChat | Discord |
| :---: | :---: |
| <img src="https://agent-one-lab.github.io/assets/agentfly/wechat.jpg" width="200" /> <br> Scan to join wechat group | <img src="https://agent-one-lab.github.io/assets/agentfly/discord.png" width="200" /> <br> [Join our discord channel](https://discord.gg/Ze5Z9QhhJ3) |
