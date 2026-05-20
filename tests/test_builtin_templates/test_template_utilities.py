from chat_bricks import get_template, register_template
from chat_bricks.templates import Template
from chat_bricks.vision import get_processor

def test_template_registration():
    register_template(
        Template(
            name="test",
            system_template="",
            user_template="",
            assistant_template="",
            stop_words=[]
        )
    )
    assert get_template("test") is not None
    assert get_template("test").name == "test"

def test_template_registration_with_vision():
    register_template(
        Template(
            name="test-vl",
            system_template="",
            user_template="",
            assistant_template="",
            stop_words=[],
            image_token="<|image_pad|>",
        )
    )
    assert get_processor("test-vl") is not None
    assert get_processor("test-vl").config.image_token == "<|image_pad|>"
