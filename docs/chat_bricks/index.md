# Chat Template System Documentation

## Quick Start

```python
from chat_bricks import Chat, get_template

# Get a pre-built template
template = get_template("qwen3")

# Create a chat instance
chat = Chat(template="qwen3", messages=[
    {"role": "user", "content": "Hello, how are you?"}
])

# Generate a prompt
prompt = chat.prompt()
print(prompt)
```

## Key Concepts

### Template Components

- **System Template**: Defines system message format. May include `{tools}` / `{skills}` slots filled by the section templates below.
- **User Template**: How user messages are formatted.
- **Assistant Template**: How assistant responses are formatted.
- **Observations Template**: How tool responses are formatted (formerly `tool_template`).
- **Tools Template** + **Single Tool Template**: Section wrappers for the tool catalogue.
- **Skills Template** + **Single Skill Template**: Section wrappers for the skill catalogue — `(name, description)` entries advertised in the system prompt.

### Policies

- **System Policy**: Controls system message behavior.
- **Tool Policy**: Manages tool placement, formatting, and content processing.
- **Skill Policy**: Controls how each skill entry is rendered into the `{skills}` block.
- **Global Policy**: Template-wide behavior settings.

### Vision Support

- **Image Processing**: Automatic image token expansion
- **Video Processing**: Video frame extraction and processing
- **Multi-Modal Alignment**: Proper tensor alignment for training

## Getting Started

1. Read [Basic Usage](basic_usage.md) for quick setup.
2. Explore [Examples](examples.md) to see practical implementations.
3. Create [Custom Templates](custom_templates.md) for your specific needs.
4. Leverage [Advanced Features](advanced_features.md) for complex use cases.
5. Add [Vision Support](vision_templates.md) for multi-modal capabilities.

## Design Philosophy

The Chat Template System is inspired by **building block toys**—complex structures are created by combining simple, standardized components. This philosophy manifests in:

- **Modularity**: Interchangeable, composable, extensible components
- **Separation of Concerns**: Each component has a single, well-defined responsibility
- **Strategy Pattern**: Different behaviors can be selected at runtime
- **Policy-Based Configuration**: Flexible behavior control without hardcoding

## System Architecture

```
Messages + Tools + Skills → Template Processing → Vision Processing → LLM-Ready Inputs
```

The system follows a **four-step rendering process**:

1. **Tool Insertion**: Decide where and how to inject the tool catalogue.
2. **Skill Formatting**: Build the skill catalogue block for the `{skills}` slot.
3. **Turn Encoding**: Convert each conversation turn to its textual representation.
4. **Generation Prompt**: Optionally append generation prefixes.

## Contributing

The template system is designed to be extensible. See [Custom Templates](custom_templates.md) for guidance on adding new template types and processors.
