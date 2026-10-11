"""``default_system_template`` formats the default system turn; a supplied system
message still uses ``system_template``."""

from chat_bricks.templates.templates import Template

TEMPLATE = Template(
    name="test-default-system-template",
    system_template="<sys>{system_message}</sys>",
    default_system_template="<sys>{system_message}</sys>\n",
    system_message="default",
    user_template="<user>{content}</user>",
    assistant_template="<asst>{content}</asst>",
)


def test_default_system_message_uses_default_system_template():
    prompt = TEMPLATE.render([{"role": "user", "content": "hi"}])[0]
    assert prompt == "<sys>default</sys>\n<user>hi</user>"


def test_supplied_system_message_uses_system_template():
    messages = [{"role": "system", "content": "mine"}, {"role": "user", "content": "hi"}]
    assert TEMPLATE.render(messages)[0] == "<sys>mine</sys><user>hi</user>"


def test_unset_default_system_template_falls_back_to_system_template():
    template = TEMPLATE.copy()
    template.default_system_template = None
    prompt = template.render([{"role": "user", "content": "hi"}])[0]
    assert prompt == "<sys>default</sys><user>hi</user>"


def test_copy_preserves_default_system_template():
    assert TEMPLATE.copy().default_system_template == TEMPLATE.default_system_template
