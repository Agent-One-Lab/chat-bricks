"""Skill rendering policy.

Mirrors :class:`ToolPolicy` but is intentionally minimal: a skill is just a
``name`` + ``description`` per the skill spec, so the policy only needs a
single per-skill template and a join behaviour. Skills always live in the
system message — there is no equivalent of ``ToolPlacement``.
"""

import dataclasses
from typing import Any, Callable, Dict, List, Mapping


def _coerce_skill(skill: Any) -> Dict[str, Any]:
    """Accept dicts or objects exposing ``name``/``description`` attributes."""
    if isinstance(skill, Mapping):
        return dict(skill)
    name = getattr(skill, "name", None)
    description = getattr(skill, "description", "")
    if name is None:
        raise TypeError(
            f"skill entries must be dicts or have a .name attribute, got {type(skill).__name__}"
        )
    return {"name": name, "description": description}


@dataclasses.dataclass
class SkillPolicy:
    """How a list of skills becomes the inner text of the ``{skills}`` placeholder."""

    single_skill_template: str = "- {name}: {description}"
    joiner: str = "\n"
    content_processor: Callable[[Dict], Dict] = None

    def format_skill(self, skill: Any) -> str:
        skill_dict = _coerce_skill(skill)
        if self.content_processor is not None:
            skill_dict = self.content_processor(skill_dict)
        return self.single_skill_template.format(**skill_dict)

    def format_skills(self, skills: List[Any]) -> str:
        return self.joiner.join(self.format_skill(s) for s in skills)
