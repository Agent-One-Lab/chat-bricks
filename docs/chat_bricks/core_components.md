
<style>

.system {
  background-color: #cdb4db;
  border-radius: 6px;
  padding: 2px 6px;
}
.user {
  background-color: #ffc8dd;
  border-radius: 6px;
  padding: 2px 6px;
}
.assistant {
  background-color: #ffafcc;
  border-radius: 6px;
  padding: 2px 6px;
}
.tool {
  background-color: #bde0fe;
  border-radius: 6px;
  padding: 2px 6px;
}
</style>




# Core Components

### Core Chat Template Components

The Chat Template System is inspired by the art of building block toys - where complex structures are created by combining simple, standardized components. We identify some basic components from LLM's chat templates, and use them to form prompts from conversation messages. Below are the core components:

`system_template`: Specify how the system prompt is formatted. May contain `{system_message}`, and optionally `{tools}` / `{skills}` slots that get filled by the section templates below.

`user_template` / `user_template_with_tools`: Specify how a user message is formatted (the `_with_tools` variant is used when the tool policy places the tool catalogue with a user turn).

`assistant_template`: Specify how an assistant message is formatted.

`observations_template` (formerly `tool_template`): Wraps a tool-response message. Use `{observation}` for single responses or `{observations}` when combined with `single_observation_template` for parallel tool responses.

`tools_template` + `single_tool_template`: Section wrappers used for the tool catalogue. The renderer wraps each tool with `single_tool_template`, joins them, then wraps the whole list with `tools_template`. The result fills the `{tools}` placeholder in `system_template` (or in `user_template_with_tools` depending on tool placement).

`skills_template` + `single_skill_template`: Section wrappers for the skill catalogue. Same two-pass pattern as tools — the result fills the `{skills}` placeholder in `system_template`. Skills only live in the system message.

`tool_calls_template` + `single_tool_call_template` (formerly `tool_call_template`): Wraps parallel tool calls inside an assistant message.

Assume we have the following chat template, and messages
```python
system_template = "System: {system_message}{tools}\n"
tools_template = "\n#Tools: {tools}"
user_template = "User: {content}\n"
assistant_template = "Assistant: {content}\n"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi, Can you help me search the information."},
    {"role": "assistant", "content": "tool call: search tool arguments: related query"},
    {"role": "tool", "content": "Searched information..."}
]

tools = [
    {
        "name": "search",
        "description": "Search the web."
    }
]
```

**Formatted Prompt**

<span class="system"> </span>: formatted system prompt; <span class="user"> </span>: formatted user message; <span class="assistant"> </span>: formatted assistant message; <span class="tool"> </span>: formatted tool message;

1. When no tools are passed, the `{tools}` slot is empty and the prompt is:

<span class="system">System: You are a helpful assistant.</span>

<span class="user">User: Hi, Can you help me search the information.</span>

<span class="assistant">Assistant: tool call: search\ntool arguments: related query</span>

<span class="tool">Tool: Searched information...</span>

2. When tools are included, the `tools_template` fills the `{tools}` slot in `system_template`:

<span class="system">System: You are a helpful assistant.
<br>
#Tools: [{"name": "search", "description": "Search the web"}]</span>

<span class="user">User: Hi, Can you help me search the information.</span>

<span class="assistant">Assistant: tool call: search\ntool arguments: related query</span>

<span class="tool">Tool: Searched information...</span>


### High-Level Workflow

```
Messages + Tools + Skills → Template Processing → Vision Processing → LLM-Ready Inputs
```

The system follows a four-step rendering process:

1. **Tool Insertion**: Decide where and how to inject the tool catalogue (system message or first/last user turn).
2. **Skill Formatting**: Build the skill catalogue block that fills the `{skills}` slot of the system template (system-only, no placement variation).
3. **Turn Encoding**: Convert each conversation turn to its textual representation.
4. **Generation Prompt**: Optionally append generation prefixes.

If we tokenize the input messages, the vision processor will do the following steps:

- **Template** → Human-readable prompt with vision tokens
- **Vision Processor** → Token expansion and multi-modal inputs
- **Result** → LLM-ready inputs with proper tensor alignment

### Core Class Components

#### Template
The central class that manages:
- Message formatting templates
- Policy configurations
- Jinja template generation

#### Chat
Recommended class for user usage:
- Store and format messages
- Get formatted prompts
- Tokenize formatted prompt

### Advanced Features

**1. Register & Obtain Template**

Templates are created and retrieved through a global registry:
```python
# Registration
register_template(Template(name="custom", ...))

# Retrieval
template = get_template("custom")
```

**2. Fine-grained Behavior Control**

Four levels of policy control:

1. **Global Policy**: Template-wide settings (e.g., prefix tokens)
2. **System Policy**: System message behavior and content processing
3. **Tool Policy**: Tool placement, formatting, and content processing
4. **Skill Policy**: How a `(name, description)` skill entry becomes one row in the `{skills}` block

```python
# Tool formatting strategies
JsonFormatter(indent=4)
JsonCompactFormatter()
YamlFormatter()

# Tool placement strategies
ToolPlacement.SYSTEM
ToolPlacement.FIRST_USER
ToolPlacement.LAST_USER

# Skill row template (default: "- {name}: {description}")
from chat_bricks.policies import SkillPolicy
SkillPolicy(single_skill_template="* {name} :: {description}", joiner="\n")
```

**3. Vision Process**

Vision processors are automatically registered when vision tokens are detected:
```python
def _register_vision_processor(self):
    """Automatically register a vision processor for this template"""
    if self.image_token or self.video_token:
        # Auto-registration based on template configuration
```


**4. Jinja Template Generation**

Templates can generate HuggingFace-compatible Jinja templates:
- Enables use with external systems (vLLM, transformers tokenizers, etc.)
- Maintains consistency between Python and Jinja rendering
- Supports complex logic through Jinja macros
