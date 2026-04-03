from typing import Dict

from ..templates import HFTemplate, Template

# A global registry for all conversation templates
TEMPLATES: Dict[str, Template] = {}


def register_template(template: Template, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in TEMPLATES, f"{template.name} has been registered."

    TEMPLATES[template.name] = template


def get_template(name: str, tokenizer=None) -> Template:
    """Get a conversation template."""
    if name in TEMPLATES:
        return TEMPLATES[name].copy()
    else:
        # Use HF's tokenizer chat template
        return HFTemplate(name, tokenizer=tokenizer)


# Register built-in templates on import so get_template can find them.
from . import builtin  # noqa: F401, E402

__all__ = [
    "register_template",
    "get_template",
]
