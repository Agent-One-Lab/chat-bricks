"""Jinja parity tests for skills.

The :class:`JinjaGenerator` was updated to emit ``_skills_template`` /
``_single_skill_template`` / ``_skill_joiner`` macros that consume a ``skills``
variable passed via ``tokenizer.apply_chat_template(messages, skills=...)``.

These tests pin the contract: the Python ``Chat.prompt()`` output and the
generated Jinja template, when invoked through ``apply_chat_template`` on a real
HF tokenizer, must agree byte-for-byte across the four states (no/tools-only/
skills-only/both). A regression in either code path will produce a string diff.
"""

import pytest
from transformers import AutoTokenizer

from chat_bricks import Chat, get_template


TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as e:
        pytest.skip(f"Tokenizer {TOKENIZER_ID} not available: {e}")


MESSAGES = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "hi"},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load a skill",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    }
]

SKILLS = [
    {"name": "add-numbers", "description": "Adds two integers."},
    {"name": "word-count", "description": "Counts words in text."},
]


def _apply_via_jinja(tokenizer, template_name, **kwargs):
    """Render through the generated Jinja template using HF apply_chat_template."""
    template = get_template(template_name)
    tokenizer.chat_template = template.jinja_template()
    return tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, **kwargs
    )


def _render_via_python(template_name, **kwargs):
    """Render through the Python Renderer path."""
    return Chat(template_name, MESSAGES, **kwargs).prompt()


@pytest.mark.parametrize(
    "tools,skills",
    [
        (None, None),
        (TOOLS, None),
        (None, SKILLS),
        (TOOLS, SKILLS),
    ],
    ids=["neither", "tools-only", "skills-only", "both"],
)
def test_jinja_matches_python_for_skills_states(tokenizer, tools, skills):
    """Across all four (tools, skills) states, Jinja and Python paths must
    produce the same prompt for the ``qwen-skills`` template."""
    python_prompt = _render_via_python("qwen-skills", tools=tools, skills=skills)
    jinja_prompt = _apply_via_jinja(
        tokenizer, "qwen-skills", tools=tools, skills=skills
    )
    assert jinja_prompt == python_prompt, (
        f"Mismatch with tools={'yes' if tools else 'no'}, "
        f"skills={'yes' if skills else 'no'}.\n"
        f"--- Python ---\n{python_prompt}\n--- Jinja ---\n{jinja_prompt}"
    )


def test_jinja_template_string_contains_skill_machinery(tokenizer):
    """A quick structural check: the generated template string for a
    skills-aware template should include the ``skills`` block setup so callers
    can grep it for sanity in CI logs."""
    template = get_template("qwen-skills")
    jinja_str = template.jinja_template()
    # The machinery added by JinjaGenerator for the skills section
    assert "_skills_template" in jinja_str
    assert "_single_skill_template" in jinja_str
    assert "_skill_joiner" in jinja_str


def test_jinja_ignores_skills_for_template_without_skills_template(tokenizer):
    """A template that doesn't define ``skills_template`` should produce the
    same Jinja output whether or not ``skills=`` is passed."""
    # qwen2.5 has no skills_template.
    prompt_without = _apply_via_jinja(tokenizer, "qwen2.5")
    prompt_with = _apply_via_jinja(tokenizer, "qwen2.5", skills=SKILLS)
    assert prompt_without == prompt_with
    assert "add-numbers" not in prompt_with
