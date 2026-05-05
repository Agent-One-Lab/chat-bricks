from chat_bricks.chat import Chat


def test_chat_text_only_still_probes_vision_model_capability():
    messages = [{"role": "user", "content": "Hello from a text-only conversation."}]

    chat = Chat(template="Qwen/Qwen3.5-4B", messages=messages)

    assert chat.is_vision_template is True


def test_chat_with_vision_content_probes_vision_model():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image", "image": "/tmp/demo.png"},
            ],
        }
    ]

    chat = Chat(template="Qwen/Qwen3.5-4B", messages=messages)

    assert chat.is_vision_template is True


def test_vision_inputs_text_only_returns_empty():
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Text only, no image and no video."}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Acknowledged."}],
        },
    ]

    chat = Chat(template="Qwen/Qwen3.5-4B", messages=messages)
    is_vision_template = chat.is_vision_template
    assert is_vision_template is True

    vision_inputs = chat.vision_inputs()

    prompt = chat.prompt()

    print(prompt)

    assert len(vision_inputs) == 0
