"""Tests for the ``deepseek-v4`` builtin template.

DeepSeek-V4 (``deepseek-ai/DeepSeek-V4-Pro`` / ``-Flash``) ships a *Python*
encoder (``encoding/encoding_dsv4.py``) instead of a Jinja ``chat_template``.
chat-bricks models its **chat mode** (``thinking_mode="chat"``) structure, where
the thinking block is closed immediately with ``</think>``.

The ``GOLDEN_*`` strings below were produced by running that reference encoder in
chat mode over the model's own ``encoding/tests`` fixtures (tools attached to the
system message, as the encoder expects). They pin the DSML tool-call format and
the multi-turn structure byte-for-byte. ``Skills`` is a chat-bricks addition in
V4's markdown style — V4 has no native skills concept — so it is checked
structurally rather than against the reference.
"""

from chat_bricks import Chat


# --- Reference fixtures (reproduced from encoding/tests, chat mode) ----------

GOLDEN_BASIC = (
    "<｜begin▁of▁sentence｜>You are a helpful assistant."
    "<｜User｜>Hello<｜Assistant｜></think>Hi there! How can I help you?<｜end▁of▁sentence｜>"
    "<｜User｜>What is the capital of France?<｜Assistant｜></think>"
    "The capital of France is Paris.<｜end▁of▁sentence｜>"
)

BASIC_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there! How can I help you?"},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]

GOLDEN_TOOLS = (
    '<｜begin▁of▁sentence｜>You are a helpful assistant.\n\n## Tools\n\nYou have access '
    'to a set of tools to help answer the user\'s question. You can invoke tools by '
    'writing a "<｜DSML｜tool_calls>" block like the following:\n\n<｜DSML｜tool_calls>\n'
    '<｜DSML｜invoke name="$TOOL_NAME">\n<｜DSML｜parameter name="$PARAMETER_NAME" '
    'string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n'
    '<｜DSML｜invoke name="$TOOL_NAME2">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\n'
    'String parameters should be specified as is and set `string="true"`. For all '
    'other types (numbers, booleans, arrays, objects), pass the value in JSON format '
    'and set `string="false"`.\n\nIf thinking_mode is enabled (triggered by <think>), '
    'you MUST output your complete reasoning inside <think>...</think> BEFORE any tool '
    'calls or final response.\n\nOtherwise, output directly after </think> with tool '
    'calls or final response.\n\n### Available Tool Schemas\n\n'
    '{"name": "get_weather", "description": "Get the weather for a specific location", '
    '"parameters": {"type": "object", "properties": {"location": {"type": "string", '
    '"description": "The city name"}, "unit": {"type": "string", "enum": ["celsius", '
    '"fahrenheit"], "description": "Temperature unit"}}, "required": ["location"]}}\n'
    '{"name": "search", "description": "Search the web for information", "parameters": '
    '{"type": "object", "properties": {"query": {"type": "string", "description": '
    '"Search query"}, "num_results": {"type": "integer", "description": "Number of '
    'results to return"}}, "required": ["query"]}}\n\nYou MUST strictly follow the '
    'above defined tool name and parameter schemas to invoke tool calls.\n'
    '<｜User｜>What\'s the weather in Beijing?<｜Assistant｜></think>\n\n<｜DSML｜tool_calls>\n'
    '<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" '
    'string="true">Beijing</｜DSML｜parameter>\n<｜DSML｜parameter name="unit" '
    'string="true">celsius</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
    '<｜end▁of▁sentence｜><｜User｜><tool_result>{"temperature": 22, "condition": '
    '"sunny", "humidity": 45}</tool_result><｜Assistant｜></think>The weather in Beijing '
    'is currently sunny with a temperature of 22°C and 45% humidity.<｜end▁of▁sentence｜>'
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Beijing?"},
    {
        "role": "assistant",
        # Tool-call turn with no ``content`` key — the common OpenAI shape.
        "tool_calls": [
            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Beijing", "unit": "celsius"}',
                },
            }
        ],
    },
    {"role": "tool", "content": '{"temperature": 22, "condition": "sunny", "humidity": 45}'},
    {
        "role": "assistant",
        "content": "The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity.",
    },
]

SKILLS = [
    {"name": "pdf", "description": "Read and edit PDF files."},
    {"name": "xlsx", "description": "Work with Excel spreadsheets."},
]


# --- Reference parity --------------------------------------------------------


def test_basic_multi_turn_matches_reference():
    prompt = Chat("deepseek-v4", BASIC_MESSAGES).prompt()
    assert prompt == GOLDEN_BASIC


def test_tool_calling_matches_reference():
    """Full tool round-trip: catalogue, DSML invoke block, tool_result, answer."""
    prompt = Chat("deepseek-v4", TOOL_MESSAGES, tools=TOOLS).prompt()
    assert prompt == GOLDEN_TOOLS


def test_generation_prompt_closes_reasoning_block():
    prompt = Chat(
        "deepseek-v4",
        [{"role": "user", "content": "hi"}],
    ).prompt(add_generation_prompt=True)
    assert prompt.endswith("<｜Assistant｜></think>")


# --- DSML tool-call serialization -------------------------------------------


def test_dsml_typed_parameters():
    """String args use ``string="true"`` verbatim; non-strings are JSON with
    ``string="false"`` (mirrors ``encode_arguments_to_dsml``)."""
    messages = [
        {"role": "user", "content": "search"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "cats", "num_results": 5, "safe": true}',
                    },
                }
            ],
        },
    ]
    prompt = Chat("deepseek-v4", messages, tools=TOOLS).prompt()
    assert '<｜DSML｜parameter name="query" string="true">cats</｜DSML｜parameter>' in prompt
    assert '<｜DSML｜parameter name="num_results" string="false">5</｜DSML｜parameter>' in prompt
    assert '<｜DSML｜parameter name="safe" string="false">true</｜DSML｜parameter>' in prompt


def test_parallel_tool_calls_joined_in_one_block():
    messages = [
        {"role": "user", "content": "two things"},
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": '{"location": "A"}'}},
                {"type": "function", "function": {"name": "search", "arguments": '{"query": "B"}'}},
            ],
        },
    ]
    prompt = Chat("deepseek-v4", messages, tools=TOOLS).prompt()
    # Scope to the generated assistant turn — the ``## Tools`` documentation block
    # also contains literal ``<｜DSML｜tool_calls>`` examples.
    generated = prompt.rsplit("<｜Assistant｜></think>", 1)[1]
    # Exactly one wrapping block holding both invokes.
    assert generated.count("<｜DSML｜tool_calls>") == 1
    assert generated.count("</｜DSML｜tool_calls>") == 1
    assert generated.count("<｜DSML｜invoke") == 2
    assert 'name="get_weather"' in generated and 'name="search"' in generated


# --- Skills (chat-bricks addition) ------------------------------------------


def test_skills_only_renders_skills_block():
    prompt = Chat("deepseek-v4", BASIC_MESSAGES[:2], skills=SKILLS).prompt()
    assert "## Skills" in prompt
    assert "### Available Skills" in prompt
    assert "- pdf: Read and edit PDF files." in prompt
    assert "- xlsx: Work with Excel spreadsheets." in prompt
    assert "## Tools" not in prompt


def test_tools_and_skills_render_in_order():
    prompt = Chat("deepseek-v4", BASIC_MESSAGES[:2], tools=TOOLS, skills=SKILLS).prompt()
    assert "## Tools" in prompt and "## Skills" in prompt
    assert prompt.index("## Tools") < prompt.index("## Skills")


def test_empty_skills_list_renders_no_skill_block():
    prompt = Chat("deepseek-v4", BASIC_MESSAGES[:2], skills=[]).prompt()
    assert "## Skills" not in prompt


# --- Training mask -----------------------------------------------------------


def test_mask_marks_tool_call_and_answer_as_trainable():
    """The DSML tool-call block and the final answer must be on the assistant
    (trainable) side of the mask; prompt, catalogue and tool_result must not."""
    chat = Chat("deepseek-v4", TOOL_MESSAGES, tools=TOOLS)
    # render() returns (prompt, elements, mask_flags); mask_flag True == masked.
    _, elements, mask_flags, _ = chat.template.render(messages=chat.messages, tools=TOOLS)
    trainable = "".join(e for e, masked in zip(elements, mask_flags) if not masked)
    masked = "".join(e for e, masked in zip(elements, mask_flags) if masked)

    assert '<｜DSML｜invoke name="get_weather">' in trainable
    assert "The weather in Beijing is currently sunny" in trainable
    assert "## Tools" in masked
    assert "<tool_result>" in masked
    assert "What's the weather in Beijing?" in masked
