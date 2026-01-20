from chat_bricks.templates import HFTemplate
from chat_bricks.chat import Chat
import pytest
from transformers import AutoTokenizer
from chat_bricks.utils.process import strip_ansi

@pytest.mark.parametrize("repo_name", ["Qwen/Qwen2.5-3B-Instruct", "moonshotai/Kimi-K2-Instruct"])
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
def test_hf_templates(repo_name, messages, tools, add_generation_prompt):
    chat = Chat(repo_name, messages=messages)
    prompt = chat.prompt(add_generation_prompt=add_generation_prompt, tools=tools)
    prompt_with_mask = chat.prompt_with_mask(add_generation_prompt=add_generation_prompt, tools=tools)

    raw_prompt = strip_ansi(prompt_with_mask)

    tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
    hf_template = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, add_generation_prompt=add_generation_prompt)
    
    is_equal = prompt == hf_template

    is_equal_between_implemented_and_constructed_prompt = prompt == raw_prompt

    if not is_equal:
        print(f"Implemented prompt:\n\n{prompt}")
        print(f"HF template prompt:\n\n{hf_template}")
    assert is_equal, f"Implemented prompt:\n\n{prompt}\n\nHF template prompt:\n\n{hf_template}"

    if not is_equal_between_implemented_and_constructed_prompt:
        print(f"Implemented prompt:\n\n{prompt}")
        print(f"Constructed prompt:\n\n{raw_prompt}")
    assert is_equal_between_implemented_and_constructed_prompt, f"Implemented prompt:\n\n{prompt}\n\nConstructed prompt:\n\n{raw_prompt}"