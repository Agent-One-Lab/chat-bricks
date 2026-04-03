import re
from typing import Any, List
import copy

from ..vision import get_processor

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")  # matches any ANSI color/style code


def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return ANSI_RE.sub("", s)


def convert_messages_to_hf_format(messages: list) -> list:
    """
    Convert messages to Hugging Face format.
    """
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            for item in content:
                if "type" in item:
                    if item["type"] == "image_url":
                        item["type"] = "image"
                        item["image"] = item["image_url"]["url"]
                        del item["image_url"]
                    else:
                        # TODO: handle other types of content
                        pass
        message["content"] = content
    return messages


def convert_inputs_to_vision_inputs(
    template: str,
    inputs: dict,
    processor,  # AutoProcessor (not bare tokenizer)
    messages: list,
):
    """
    NEW PIPELINE: Template processes messages → Human-readable prompt → Vision processor → LLM-ready inputs

    The correct pipeline is:
    1. Template processes messages to get human-readable prompt with single multi-modal tokens
    2. Vision processor handles image/video processing and token expansion
    3. Final result is directly usable by LLMs with model(**inputs)
    """
    # Get the vision processor for this template
    vision_processor = get_processor(template)
    if vision_processor is None:
        raise ValueError(f"No vision processor registered for template: {template}")

    # Step 1: Template processes messages to get human-readable prompt
    from .. import Chat

    chat = Chat(template=template, messages=messages, tokenizer=processor.tokenizer)
    prompt = (
        chat.prompt()
    )  # This gives us human-readable prompt with single multi-modal tokens

    # Step 2: Extract vision inputs from messages
    images, videos = extract_vision_inputs_from_messages(messages)

    # Step 3: Vision processor handles the complete pipeline
    # This expands tokens and generates LLM-ready inputs
    final_inputs = vision_processor.process_for_llm(
        prompt=prompt,
        images=images,
        videos=videos,
        processor=processor,
        tokenizer=processor.tokenizer,
    )

    return final_inputs


def extract_vision_inputs_from_messages(messages: List) -> tuple[List, List]:
    """Extract images and videos from messages"""
    images, videos = [], []

    for message in messages:
        if isinstance(message.get("content"), List):
            for item in message["content"]:
                if item.get("type") in ["image", "image_url"]:
                    if "image" in item:
                        images.append(item["image"])
                    elif "image_url" in item:
                        images.append(item["image_url"]["url"])
                elif item.get("type") in ["video", "video_url"]:
                    if "video" in item:
                        videos.append(item["video"])
                    elif "video_url" in item:
                        videos.append(item["video_url"]["url"])

    return images, videos


def process_prompt_with_vision(
    prompt: str,
    template: str,
    processor: Any,
    images: list = None,
    videos: list = None,
) -> dict:
    """Process a prompt with vision support"""
    vision_processor = get_processor(template)
    if vision_processor is None:
        # If no vision processor, just return tokenized prompt
        return processor.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
        )

    # Use vision processor to handle the complete pipeline
    return vision_processor.process_for_llm(
        prompt=prompt,
        images=images or [],
        videos=videos or [],
        processor=processor,
        tokenizer=processor.tokenizer,
    )


def visualize_template(template, messages=None, tools=None, **kwargs):
    from .. import Chat

    if not messages:
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am fine, thank you."},
            {"role": "user", "content": "Want to play a game?"},
            {"role": "assistant", "content": "Sure, what game?"},
            {"role": "user", "content": "Guess the number."},
        ]

    chat = Chat(template=template, messages=messages)
    print(chat.prompt(tools=tools))
    print(chat.prompt_with_mask(tools=tools))


def visualize_jinja_template(tokenizer, messages=None, tools=None, **kwargs):
    if not messages:
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am fine, thank you."},
            {"role": "user", "content": "Want to play a game?"},
            {"role": "assistant", "content": "Sure, what game?"},
            {"role": "user", "content": "Guess the number."},
        ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, tools=tools, **kwargs
    )
    print(prompt)


def compare_hf_template(
    tokenizer,
    template_name,
    messages=None,
    tools=None,
    add_generation_prompt=False,
    **kwargs,
):
    from .. import Chat

    official_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    chat = Chat(template_name, messages=messages, tokenizer=tokenizer)
    implemented_prompt = chat.prompt(
        add_generation_prompt=add_generation_prompt, tools=tools, **kwargs
    )
    is_equal = official_prompt == implemented_prompt
    highlighted_prompt = chat.prompt_with_mask(
        add_generation_prompt=add_generation_prompt, tools=tools, **kwargs
    )
    plain_highlighted_prompt = strip_ansi(highlighted_prompt)
    is_equal_between_implemented_prompts = (
        implemented_prompt == plain_highlighted_prompt
    )
    jinja_template = chat.template.jinja_template()

    official_jinja_prompt = tokenizer.chat_template
    tokenizer.chat_template = jinja_template
    implemented_jinja_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    is_equal_between_jinja_prompts = implemented_jinja_prompt == implemented_prompt
    tokenizer.chat_template = official_jinja_prompt
    return (
        is_equal,
        is_equal_between_implemented_prompts,
        is_equal_between_jinja_prompts,
        official_prompt,
        implemented_prompt,
        implemented_jinja_prompt,
        highlighted_prompt,
    )


def compare_hf_template_with_renderer(
    repo_name,
    template_name,
    messages=None,
    tools=None,
    add_generation_prompt=False,
    **kwargs,
):
    from .. import Chat

    chat = Chat(template_name, messages=messages)
    chat_hf_template = Chat(repo_name, messages=messages)

    implemented_prompt = chat.prompt(
        add_generation_prompt=add_generation_prompt, tools=tools, **kwargs
    )
    implemented_hf_template_prompt = chat_hf_template.prompt(
        add_generation_prompt=add_generation_prompt, tools=tools, **kwargs
    )
    is_equal = implemented_prompt == implemented_hf_template_prompt
    highlighted_prompt = chat.prompt_with_mask(
        add_generation_prompt=add_generation_prompt, tools=tools, **kwargs
    )
    highlighted_hf_template_prompt = chat_hf_template.prompt_with_mask(
        add_generation_prompt=add_generation_prompt, tools=tools, **kwargs
    )
    is_equal_between_implemented_and_highlighted_hf_template_prompts = (
        implemented_prompt == highlighted_hf_template_prompt
    )
    return (
        is_equal,
        is_equal_between_implemented_and_highlighted_hf_template_prompts,
        implemented_prompt,
        implemented_hf_template_prompt,
        highlighted_prompt,
        highlighted_hf_template_prompt,
    )


def validate_messages_for_template(
    template_name, messages, tools=None, add_generation_prompt=False
):
    """Validate the messages for the given template."""
    from .. import get_template

    if add_generation_prompt and messages[-1]["role"] == "assistant":
        return False

    template = get_template(template_name)

    if tools and not template._supports_tool_call():
        return False

    if template_name in ["llama-3.2"]:
        for message in messages:
            if "tool_calls" in message and len(message["tool_calls"]) > 1:
                return False

    return True

def split_messages_with_assistant(messages: List) -> List:
    """Split messages with assistant"""
    splited_messages_list = []
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            splited_messages_list.append(copy.deepcopy(messages[:i+1]))
    return splited_messages_list
    