"""Tests for skill rendering — new SkillPolicy + section-template pattern.

Covers the four states (none / tools-only / skills-only / both), object vs dict
skill entries, custom ``single_skill_template``, and that a template without a
``skills_template`` ignores the ``skills=`` argument cleanly.
"""

from dataclasses import dataclass

import pytest

from chat_bricks import Chat, Template, register_template
from chat_bricks.policies import SkillPolicy


SYSTEM_MESSAGES = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "hi"},
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load a skill",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
    },
]
SKILLS = [
    {"name": "add-numbers", "description": "Adds two integers."},
    {"name": "word-count", "description": "Counts words in text."},
]


def test_qwen_skills_neither_renders_plain_system():
    prompt = Chat("qwen-skills", SYSTEM_MESSAGES).prompt()
    print(f"prompt: {prompt}")
    assert "You are an agent." in prompt
    assert "# Tools" not in prompt
    assert "# Skills" not in prompt
    assert "<tools>" not in prompt
    assert "<skills>" not in prompt


def test_qwen_skills_tools_only_renders_tools_block():
    prompt = Chat("qwen-skills", SYSTEM_MESSAGES, tools=TOOLS).prompt()
    print(f"prompt: {prompt}")
    assert "# Tools" in prompt
    assert "<tools>" in prompt and "</tools>" in prompt
    assert "load_skill" in prompt
    assert "# Skills" not in prompt
    assert "<skills>" not in prompt


def test_qwen_skills_skills_only_renders_skills_block():
    prompt = Chat("qwen-skills", SYSTEM_MESSAGES, skills=SKILLS).prompt()
    print(f"prompt: {prompt}")
    assert "# Skills" in prompt
    assert "<skills>" in prompt and "</skills>" in prompt
    assert "add-numbers: Adds two integers." in prompt
    assert "word-count: Counts words in text." in prompt
    assert "# Tools" not in prompt


def test_qwen_skills_both_renders_both_blocks_in_order():
    prompt = Chat("qwen-skills", SYSTEM_MESSAGES, tools=TOOLS, skills=SKILLS).prompt()
    print(f"prompt: {prompt}")
    assert "# Tools" in prompt and "# Skills" in prompt
    # Tools section comes before skills section in the template body
    assert prompt.index("# Tools") < prompt.index("# Skills")
    assert "load_skill" in prompt
    assert "add-numbers" in prompt


def test_skills_accept_attribute_objects():
    @dataclass
    class Skill:
        name: str
        description: str

    skills = [Skill("hello", "greet"), Skill("bye", "farewell")]
    prompt = Chat("qwen-skills", SYSTEM_MESSAGES, skills=skills).prompt()
    print(f"prompt: {prompt}")
    assert "- hello: greet" in prompt
    assert "- bye: farewell" in prompt


def test_skills_object_missing_name_raises():
    @dataclass
    class Broken:
        description: str

    with pytest.raises(TypeError, match=".name"):
        Chat("qwen-skills", SYSTEM_MESSAGES, skills=[Broken("x")]).prompt()


def test_custom_single_skill_template_overrides_policy_default():
    register_template(
        Template(
            name="qwen-skills-custom-row",
            system_template="<|im_start|>system\n{system_message}{tools}{skills}<|im_end|>\n",
            skills_template="\n<skills>\n{skills}\n</skills>",
            single_skill_template="* {name} :: {description}",
            user_template="<|im_start|>user\n{content}<|im_end|>\n",
            assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
            stop_words=["<|im_end|>"],
        )
    )
    prompt = Chat("qwen-skills-custom-row", SYSTEM_MESSAGES, skills=SKILLS).prompt()
    print(f"prompt: {prompt}")
    assert "* add-numbers :: Adds two integers." in prompt
    assert "* word-count :: Counts words in text." in prompt


def test_skill_policy_join_and_format_directly():
    policy = SkillPolicy()
    out = policy.format_skills([{"name": "a", "description": "first"}, {"name": "b", "description": "second"}])
    assert out == "- a: first\n- b: second"


def test_skills_arg_ignored_when_template_has_no_skills_template():
    """A legacy template (no ``skills_template``) must not break when skills=... is passed."""
    register_template(
        Template(
            name="qwen-no-skills-support",
            system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
            user_template="<|im_start|>user\n{content}<|im_end|>\n",
            assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
            stop_words=["<|im_end|>"],
        )
    )
    prompt = Chat("qwen-no-skills-support", SYSTEM_MESSAGES, skills=SKILLS).prompt()
    print(f"prompt: {prompt}")
    assert "add-numbers" not in prompt  # silently dropped
    assert "You are an agent." in prompt


def test_empty_skills_list_renders_no_skill_block():
    prompt = Chat("qwen-skills", SYSTEM_MESSAGES, skills=[]).prompt()
    print(f"prompt: {prompt}")
    assert "# Skills" not in prompt
    assert "<skills>" not in prompt


def test_tool_loading_with_single_tool_template_wraps_each():
    # The qwen-skills template uses single_tool_template="\n{tool}", so two tools
    # produce two lines inside <tools>...</tools>.
    second_tool = {
        "type": "function",
        "function": {"name": "read_skill_file", "description": "Read a file from a skill", "parameters": {"type": "object", "properties": {}}},
    }
    prompt = Chat("qwen-skills", SYSTEM_MESSAGES, tools=[TOOLS[0], second_tool]).prompt()
    print(f"prompt: {prompt}")
    assert "load_skill" in prompt
    assert "read_skill_file" in prompt
    # Each tool entry sits on its own line inside the wrapping block. The template
    # contains a literal "<tools></tools>" marker phrase before the real wrapping
    # pair, so use rfind to skip past the marker.
    opening = prompt.rfind("<tools>")
    closing = prompt.rfind("</tools>")
    inner = prompt[opening + len("<tools>") : closing]
    # single_tool_template="\n{tool}" → each tool entry is preceded by "\n{".
    assert inner.count("\n{") == 2
