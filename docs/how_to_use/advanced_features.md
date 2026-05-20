# Advanced Features

## Overview

The Chat Template System provides advanced features for fine-grained control over template behavior, tool integration, and system message processing. This guide covers the sophisticated capabilities that make the system powerful and flexible.

## Tool Policy System

### Tool Placement Strategies

The system supports multiple strategies for where and how tools are integrated into prompts:

```python
from chat_bricks import ToolPlacement

# 1. SYSTEM placement - tools appear in system message
system_placement = ToolPlacement.SYSTEM

# 2. FIRST_USER placement - tools appear with first user message
first_user_placement = ToolPlacement.FIRST_USER

# 3. LAST_USER placement - tools appear with last user message
last_user_placement = ToolPlacement.LAST_USER

```

### Tool Formatters

Different strategies for formatting tool definitions:

#### JSON Formatters

```python
from chat_bricks import (
    JsonFormatter, JsonMinifiedFormatter, JsonIndentedFormatter, JsonCompactFormatter
)

# Standard JSON formatter with configurable options
json_formatter = JsonFormatter(
    indent=2,                    # Pretty-printed with 2-space indent
    separators=(",", ":"),       # Standard JSON separators
    joiner="\n\n",              # Join multiple tools with double newlines
    format_as_list=False         # Format as individual JSON objects
)

# Minified JSON (single line, no whitespace)
minified_formatter = JsonMinifiedFormatter(joiner="\n")

# Indented JSON (pretty-printed)
indented_formatter = JsonIndentedFormatter(indent=4, joiner="\n\n")

# Compact JSON (single array)
compact_formatter = JsonCompactFormatter(format_as_list=True)
```

#### YAML Formatter

```python
from chat_bricks import YamlFormatter

# YAML formatting (requires PyYAML)
yaml_formatter = YamlFormatter()
```

#### Custom Formatters

```python
from chat_bricks import ToolFormatter, ToolPolicy, ToolPlacement

class CustomToolFormatter(ToolFormatter):
    def format(self, tools):
        """Custom formatting logic"""
        formatted_tools = []
        for tool in tools:
            if "function" in tool:
                func = tool["function"]
                formatted = f"Function: {func['name']}\n"
                formatted += f"Description: {func['description']}\n"
                formatted += f"Parameters: {func['parameters']}\n"
                formatted_tools.append(formatted)
        return "\n---\n".join(formatted_tools)

    def jinja(self):
        """Jinja template for custom formatting"""
        return """{% for tool in tools %}
Function: {{ tool.function.name }}
Description: {{ tool.function.description }}
Parameters: {{ tool.function.parameters }}
{% if not loop.last %}---{% endif %}
{% endfor %}"""

# Use custom formatter
custom_tool_policy = ToolPolicy(
    placement=ToolPlacement.SYSTEM,
    formatter=CustomToolFormatter()
)
```

### Tool Content Processors

Process tool content before formatting:

```python
from chat_bricks import ToolContentProcessor, ToolPolicy, ToolPlacement, JsonIndentedFormatter

class ToolFilterProcessor(ToolContentProcessor):
    """Filter tools based on certain criteria"""

    def __init__(self, allowed_categories=None):
        self.allowed_categories = allowed_categories or []

    def __call__(self, tool):
        """Filter tool based on category"""
        if not self.allowed_categories:
            return tool

        # Check if tool has category and it's allowed
        tool_category = tool.get("category", "general")
        if tool_category in self.allowed_categories:
            return tool
        return None

    def jinja(self):
        """Jinja template for filtering"""
        return "{{ tool if tool.category in ['allowed_category1', 'allowed_category2'] else none }}"

# Use content processor
filtered_tool_policy = ToolPolicy(
    placement=ToolPlacement.SYSTEM,
    formatter=JsonIndentedFormatter(),
    content_processor=ToolFilterProcessor(["search", "calculation"])
)
```

## Skill Policy System

Skills are a lightweight catalogue concept — each skill has a `name` and
`description`, and the list is advertised in the system prompt (typically next
to a `load_skill` tool). Unlike tools, skills always live in the system message
— there is no placement variation.

### How Skills Render

Three pieces decide what the skill block looks like:

1. **`{skills}` placeholder** in `system_template` — where the block lives.
2. **`skills_template`** — wraps the joined list, e.g. `"# Skills\n<skills>\n{skills}\n</skills>"`.
3. **`single_skill_template`** (or `SkillPolicy.single_skill_template`) — wraps
   one entry, defaulting to `"- {name}: {description}"`.

If `skills_template` is `None`, the template doesn't render skills and a
`skills=` argument at render time is silently dropped.

### SkillPolicy

```python
from chat_bricks import Template
from chat_bricks.policies import SkillPolicy

# Default policy — one entry per line, "- name: description"
default_skill_policy = SkillPolicy()

# Custom row format and a different joiner
custom_skill_policy = SkillPolicy(
    single_skill_template="* {name} :: {description}",
    joiner="\n",
)

# With a content processor (e.g. truncating long descriptions)
def truncate_description(skill, limit=80):
    desc = skill.get("description", "")
    if len(desc) > limit:
        skill = {**skill, "description": desc[: limit - 1] + "…"}
    return skill

policy_with_processor = SkillPolicy(content_processor=truncate_description)

template = Template(
    name="my-skills",
    system_template="<|im_start|>system\n{system_message}{skills}<|im_end|>\n",
    skills_template="\n\n# Skills\n<skills>\n{skills}\n</skills>",
    skill_policy=custom_skill_policy,
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    stop_words=["<|im_end|>"],
)
```

### Skill Entries

Skill entries may be plain dicts or any object that exposes `.name` and
`.description` attributes:

```python
from dataclasses import dataclass

@dataclass
class Skill:
    name: str
    description: str

skills = [
    {"name": "add-numbers", "description": "Adds two integers."},
    Skill("word-count", "Counts words in text."),
]

chat = Chat(template="my-skills", messages=messages, skills=skills)
print(chat.prompt())
```

Missing `name` raises `TypeError` — `description` defaults to empty if absent.

## System Policy System

### System Message Control

Fine-grained control over system message behavior:

```python
from chat_bricks import SystemPolicy

# Basic system policy
basic_policy = SystemPolicy(
    use_system=True,                           # Always include system message
    use_system_without_system_message=True,    # Include system even without explicit message
    content_processor=None                     # No content processing
)

# Conditional system policy
conditional_policy = SystemPolicy(
    use_system=True,                           # Include system when present
    use_system_without_system_message=False,   # Don't include system if no message
    content_processor=None
)

# No system policy
no_system_policy = SystemPolicy(
    use_system=False,                          # Never include system message
    use_system_without_system_message=False,
    content_processor=None
)
```

### System Content Processors

Transform system messages before rendering:

#### Built-in Processors

```python
from chat_bricks import Llama32DateProcessor

# Llama 3.2 date processor (adds current date)
llama_date_policy = SystemPolicy(
    use_system=True,
    use_system_without_system_message=True,
    content_processor=Llama32DateProcessor()
)
```

#### Custom Content Processors

```python
from chat_bricks import SystemContentProcessor, SystemPolicy

class EnvironmentAwareProcessor(SystemContentProcessor):
    """Add environment information to system messages"""

    def __init__(self, environment="production"):
        self.environment = environment

    def __call__(self, system_message):
        """Add environment context"""
        env_info = f"[Environment: {self.environment}]"
        return f"{env_info}\n{system_message}"

    def jinja(self):
        """Jinja template for environment awareness"""
        return """[Environment: {{ environment | default('production') }}]
{{ system_message }}"""

# Use custom processor
env_policy = SystemPolicy(
    use_system=True,
    use_system_without_system_message=True,
    content_processor=EnvironmentAwareProcessor("development")
)
```

#### Lambda Processors

```python
# Simple lambda processor
lambda_policy = SystemPolicy(
    use_system=True,
    use_system_without_system_message=True,
    content_processor=lambda msg: f"[{datetime.now().strftime('%H:%M')}] {msg}"
)
```

## Global Policy Configuration

### Template-Wide Settings

```python
from chat_bricks import GlobalPolicy

# Add prefix to all prompts
prefix_policy = GlobalPolicy(prefix="<|begin_of_text|>")
```

## Advanced Template Features

### Conditional Templates

The `{tools}` and `{skills}` placeholders in `system_template` expand to empty
strings when no tools/skills are passed, so one template handles both the bare
and section-enabled cases without an `_with_tools` variant.

```python
from chat_bricks import Template

conditional_template = Template(
    name="conditional",
    system_template="You are a helpful assistant.{tools}",
    tools_template=" with tools: {tools}",
    user_template="User: {content}",
    user_template_with_tools="User: {content}\n\nAvailable tools: {tools}",
    assistant_template="Assistant: {content}",
    observations_template="Tool: {observation}"
)
```

### Dynamic Template Generation

```python
def create_dynamic_template(base_name, capabilities):
    """Create template based on capabilities"""

    # Base templates
    system_base = "You are a helpful assistant."
    user_base = "User: {content}"
    assistant_base = "Assistant: {content}"

    # Add tool support if needed
    if "tools" in capabilities:
        system_base += "\n\nYou have access to tools: {tools}"
        user_base += "\n\nAvailable tools: {tools}"

    # Add vision support if needed
    if "vision" in capabilities:
        system_base += "\n\nYou can process images and videos."

    return Template(
        name=f"{base_name}-{'-'.join(capabilities)}",
        system_template=system_base,
        user_template=user_base,
        assistant_template=assistant_base,
        vision_start="<vision>" if "vision" in capabilities else None,
        vision_end="</vision>" if "vision" in capabilities else None,
        image_token="<image>" if "vision" in capabilities else None,
        video_token="<video>" if "vision" in capabilities else None
    )

# Create specialized templates
coding_template = create_dynamic_template("coding", ["tools"])
vision_template = create_dynamic_template("vision", ["vision"])
full_template = create_dynamic_template("full", ["tools", "vision"])
```

## Policy Composition and Inheritance

### Combining Policies

```python
from chat_bricks import ToolPolicy, ToolPlacement, JsonIndentedFormatter
# Create a base tool policy
base_tool_policy = ToolPolicy(
    placement=ToolPlacement.SYSTEM,
    formatter=JsonIndentedFormatter(indent=2)
)

# Create a specialized tool policy
specialized_tool_policy = ToolPolicy(
    placement=ToolPlacement.FIRST_USER,  # Override placement
    formatter=base_tool_policy.formatter, # Inherit formatter
    content_processor=ToolFilterProcessor(["search"])  # Add processor
)
```

### Template Policy Inheritance

```python
# Base template with policies
base_template = Template(
    name="base",
    system_template="Base: {system_message}",
    user_template="User: {content}",
    assistant_template="Assistant: {content}",
    tool_policy=ToolPolicy(
        placement=ToolPlacement.SYSTEM,
        formatter=JsonFormatter()
    ),
    system_policy=SystemPolicy(
        use_system=True,
        use_system_without_system_message=True
    )
)

# Specialized template inheriting policies
specialized_template = Template(
    name="specialized",
    system_template="Specialized: {system_message}",
    user_template=base_template.user_template,
    assistant_template=base_template.assistant_template,
    tool_policy=base_template.tool_policy,      # Inherit tool policy
    system_policy=base_template.system_policy   # Inherit system policy
)
```

## Advanced Tool Integration

### Tool Validation

```python
from chat_bricks import ToolPolicy, ToolPlacement, JsonFormatter

class ToolValidator:
    """Validate tool definitions"""

    @staticmethod
    def validate_tool(tool):
        """Validate a single tool"""
        required_fields = ["function", "name", "parameters"]

        if "function" in tool:
            func = tool["function"]
            for field in required_fields:
                if field not in func:
                    raise ValueError(f"Missing required field: {field}")
        else:
            for field in required_fields:
                if field not in tool:
                    raise ValueError(f"Missing required field: {field}")

        return tool

# Use validator in tool policy
validated_tool_policy = ToolPolicy(
    placement=ToolPlacement.SYSTEM,
    formatter=JsonFormatter(),
    content_processor=ToolValidator.validate_tool
)
```

## Best Practices for Advanced Features

### 1. **Policy Design**
- Keep policies focused and single-purpose
- Use composition over inheritance when possible
- Document policy behavior clearly

### 2. **Tool Integration**
- Choose appropriate placement strategies for your use case
- Validate tool definitions before processing
- Use content processors for tool transformation

### 3. **System Message Management**
- Use content processors for dynamic system content
- Consider when system messages should appear
- Balance flexibility with consistency

### 4. **Performance Considerations**
- Cache frequently used templates and policies
- Use lazy evaluation for expensive operations
- Profile template rendering performance

### 5. **Testing Advanced Features**
- Test policy combinations thoroughly
- Verify tool placement strategies work correctly
- Test content processors with various inputs

This advanced features guide covers the sophisticated capabilities that make the Chat Template System powerful and flexible. Use these features to create highly customized templates that meet your specific requirements.
