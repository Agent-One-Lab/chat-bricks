from typing import Dict
from ..templates import Template


# A global registry for all conversation templates
TEMPLATES: Dict[str, Template] = {}


def register_template(template: Template, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in TEMPLATES
        ), f"{template.name} has been registered."

    TEMPLATES[template.name] = template


def get_template(name: str) -> Template:
    """Get a conversation template."""
    return TEMPLATES[name].copy()


# Register built-in templates on import so get_template can find them.
# noqa: F401 keeps linters happy about the unused import.
from . import builtin  # pylint: disable=wrong-import-position, unused-import
