from chat_bricks.templates import HFTemplate
from chat_bricks.chat import Chat
import pytest
from chat_bricks.utils import compare_hf_template_with_renderer

@pytest.mark.parametrize("template_and_repo_name", [("qwen2.5", "Qwen/Qwen2.5-3B-Instruct")])
@pytest.mark.parametrize("messages", [
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
        {"role": "user", "content": "Want to play a game?"},
        {"role": "assistant", "content": "Sure, what game?"},
    ],
    [
        {"role": "user", "content": "Help me to calculate 3 times 5."},
        {"role": "assistant", "content": '''{"name": "multiply", "arguments": {"x": 3, "y": 5}}'''},
        {"role": "tool", "content": "15"},
    ],
    [
        {"role": "user", "content": "Help me to calculate 3 times 5."},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "addition", "arguments": {"x": 3, "y": 5}}}]},
        {"role": "tool", "content": "The answer is 15"}
    ],
    [
        {"role": "user", "content": "Help me to calculate 3 times 5."},
        {"role": "assistant", "content": "none", "tool_calls": [{"type": "function", "function": {"name": "multiply", "arguments": {"x": 3, "y": 5}}}, {"type": "function", "function": {"name": "addition", "arguments": {"x": 3, "y": 5}}}]},
        {"role": "tool", "content": "The answer is 15"},
        {"role": "tool", "content": "The answer is 8"},
    ],
    # [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Hello, how are you?"},
    #     {"role": "assistant", "content": "I am fine, thank you."},
    #     {"role": "user", "content": "What is 3 times 5?"},
    # ],
])
@pytest.mark.parametrize("tools", [
    None,
    [
        {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
        {"type": "function", "function": {"name": "addition", "description": "A function that adds two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to add"}, "y": {"type": "number", "description": "The second number to add"}}, "required": ["x", "y"]}}},
    ]
])
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_hf_templates(template_and_repo_name, messages, tools, add_generation_prompt):
    is_equal, is_equal_between_implemented_and_highlighted_hf_template_prompts, implemented_prompt, implemented_hf_template_prompt, highlighted_prompt, highlighted_hf_template_prompt = compare_hf_template_with_renderer(
        template_and_repo_name[1], template_and_repo_name[0], messages=messages, tools=tools, add_generation_prompt=add_generation_prompt)
    if not is_equal:
        print(f"Implemented prompt:\n\n{implemented_prompt}")
        print(f"Implemented HF template prompt:\n\n{implemented_hf_template_prompt}")
    if not is_equal_between_implemented_and_highlighted_hf_template_prompts:
        print(f"Highlighted prompt:\n\n{highlighted_prompt}")
        print(f"Highlighted HF template prompt:\n\n{highlighted_hf_template_prompt}")
    assert is_equal, f"Template: {template_and_repo_name[0]}\n\nMessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}\n\nImplemented HF template prompt:\n\n{implemented_hf_template_prompt}\n\nHighlighted prompt:\n\n{highlighted_prompt}\n\nHighlighted HF template prompt:\n\n{highlighted_hf_template_prompt}"
    # assert is_equal_between_implemented_and_highlighted_hf_template_prompts, f"Template: {template_and_repo_name[0]}\n\nMessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nImplemented prompt:\n\n{implemented_prompt}\n\nImplemented HF template prompt:\n\n{implemented_hf_template_prompt}\n\nHighlighted prompt:\n\n{highlighted_prompt}\n\nHighlighted HF template prompt:\n\n{highlighted_hf_template_prompt}"
