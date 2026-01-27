from chat_bricks.utils import tokenize_conversation
import pytest
from transformers import AutoTokenizer
import torch
from chat_bricks import Chat

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
        {"role": "assistant", "content": "none", "tool_calls": [{"type": "function", "function": {"name": "multiply", "arguments": {"x": 3, "y": 5}}}, {"type": "function", "function": {"name": "addition", "arguments": {"x": 3, "y": 5}}}]},
        {"role": "tool", "content": "The answer is 15"},
        {"role": "tool", "content": "The answer is 8"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
        {"role": "user", "content": "What is 3 times 5?"},
        {"role": "assistant", "content": "15"},
        {"role": "user", "content": "OK, what is 3 times 6?"},
        {"role": "assistant", "content": "18"},
    ],
])
@pytest.mark.parametrize("tools", [
    None,
    [
        {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
        {"type": "function", "function": {"name": "multiply", "description": "A function that multiplies two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The first number to multiply"}, "y": {"type": "number", "description": "The second number to multiply"}}, "required": ["x", "y"]}}},
    ]
])
@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_template_tokenize(template_and_repo_name, messages, tools, add_generation_prompt):
    tokenizer = AutoTokenizer.from_pretrained(template_and_repo_name[1], trust_remote_code=True)

    inputs = tokenize_conversation(
        messages,
        tokenizer,
        template_and_repo_name[0],
        max_length=2048,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt"
    )

    hf_template_inputs = tokenize_conversation(
        messages,
        tokenizer,
        template_and_repo_name[1],
        max_length=2048,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt"
    )

    input_ids_equal =  torch.equal(inputs["input_ids"], hf_template_inputs["input_ids"])
    if not input_ids_equal:
        print(f"Decoded implemented prompt: {tokenizer.decode(inputs['input_ids'][0])}")
        print(f"Decoded hf template prompt: {tokenizer.decode(hf_template_inputs['input_ids'][0])}")
    assert input_ids_equal, f"input_ids are not equal for template: {template_and_repo_name[0]}\n\nmessages: {messages}\n\ntools: {tools}\n\nadd_generation_prompt: {add_generation_prompt}\n\nImplemented prompt:\n\n{tokenizer.decode(inputs['input_ids'][0])}\n\nHF template prompt:\n\n{tokenizer.decode(hf_template_inputs['input_ids'][0])}"
