import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .templates import Template

logger = logging.getLogger(__name__)


class JinjaGenerator:
    def __init__(self, template: "Template"):
        self.template = template

    def generate_jinja_template(self) -> str:
        """Return a Hugging-Face style chat-template (Jinja-mini dialect).

        The implementation mirrors the structure of the `render` method in the `renderer.py` file.
        `render()` for easier maintenance:

            1.  _jinja_header_constants      – immutable `set` statements
            2.  _jinja_system_block          – first turn / system handling
            3.  _jinja_loop_messages         – remaining turns & per-role logic
            4.  _jinja_generation_block      – optional generation prefix

        Returns:
            The Jinja template string.
        """

        parts: List[str] = []

        # 1.  Constant header (always first)
        parts.extend(self._jinja_header_constants())

        # 2.  System-message handling (depends on presence of tools etc.)
        parts.extend(self._jinja_system_block())

        # 2.5 Pre-compute insert index for user placement
        parts.extend(self._jinja_compute_insert_idx())

        # 3.  Loop over remaining messages
        parts.extend(self._jinja_loop_messages())

        # 4.  Generation prefix block
        parts.extend(self._jinja_generation_block())

        template_str = "".join(parts)

        # Post-process: Replace __CURRENT_DATE__ placeholder with actual date
        if "__CURRENT_DATE__" in template_str:
            from datetime import datetime

            current_date = datetime.now().strftime("%d %b %Y")
            template_str = template_str.replace("__CURRENT_DATE__", current_date)

        return template_str

    # ------------------------------------------------------------------
    # Private helpers – keep them together for readability
    # ------------------------------------------------------------------

    def _jinja_header_constants(self) -> List[str]:
        """Return Jinja `set` statements for all constant strings."""

        # Compute default system message considering content processor. The
        # ``system_template`` may include ``{tools}`` / ``{skills}`` placeholders
        # (new section-template pattern); pass empty strings so the no-tools
        # default renders cleanly.
        if self.template.system_policy.content_processor is not None:
            processed_system_message = self.template.system_policy.content_processor(
                self.template.system_message, tools=None
            )
            default_system = self.template.system_template.format(
                system_message=processed_system_message, tools="", skills=""
            )
        else:
            default_system = self.template.system_template.format(
                system_message=self.template.system_message, tools="", skills=""
            )

        # Split templates
        try:
            u_pref, u_suff = self.template.user_template.split("{content}")
            a_pref, a_suff = self.template.assistant_template.split("{content}")
        except ValueError as exc:
            raise ValueError(
                "`user_template` / `assistant_template` must contain `{content}` placeholder"
            ) from exc

        if self.template.observations_template:
            if "{observations}" in self.template.observations_template:
                t_pref, t_suff = self.template.observations_template.split("{observations}")
            elif "{observation}" in self.template.observations_template:
                t_pref, t_suff = self.template.observations_template.split("{observation}")
            else:
                raise ValueError(
                    f"Invalid observations template: {self.template.observations_template}"
                )
        else:
            t_pref, t_suff = "", ""

        # Tokens for images / videos
        img_tok = (
            (self.template.vision_start or "")
            + (self.template.image_token or "")
            + (self.template.vision_end or "")
        )
        vid_tok = (
            (self.template.vision_start or "")
            + (self.template.video_token or "")
            + (self.template.vision_end or "")
        )

        # Check if assistant template supports tool calls
        supports_tool_calls_in_template = (
            "{tool_calls}" in self.template.assistant_template
        )

        # Check if observations template uses observations (plural) or observation (singular)
        uses_observations = (
            "{observations}" in self.template.observations_template
            if self.template.observations_template
            else False
        )

        header = [
            f"{{% set _user_pref  = {u_pref!r} %}}",
            f"{{% set _user_suff  = {u_suff!r} %}}",
            f"{{% set _assistant_pref  = {a_pref!r} %}}",
            f"{{% set _assistant_suff  = {a_suff!r} %}}",
            f"{{% set _assistant_template = {self.template.assistant_template!r} %}}",
            f"{{% set _tool_pref  = {t_pref!r} %}}",
            f"{{% set _tool_suff  = {t_suff!r} %}}",
            f"{{% set _image_token = {img_tok!r} %}}",
            f"{{% set _video_token = {vid_tok!r} %}}",
            f"{{% set _default_system = {default_system!r} %}}",
            f"{{% set _system_message = {self.template.system_message!r} %}}",
            f"{{% set _system_template = {self.template.system_template!r} %}}",
            f"{{% set _tool_placement = {self.template.tool_policy.placement.name!r} %}}",
            f"{{% set _supports_tool_calls = {supports_tool_calls_in_template} %}}",
            f"{{% set _uses_observations = {uses_observations} %}}",
        ]

        if self.template.observations_template:
            header.append(
                f"{{% set _observations_template = {self.template.observations_template!r} %}}"
            )
        else:
            header.append("{% set _observations_template = '' %}")

        # Add single_tool_call_template if it exists
        if self.template.single_tool_call_template:
            header.append(
                f"{{% set _single_tool_call_template = {self.template.single_tool_call_template!r} %}}"
            )

        # Add tool_calls_template if it exists
        if self.template.tool_calls_template:
            header.append(
                f"{{% set _tool_calls_template = {self.template.tool_calls_template!r} %}}"
            )
        else:
            header.append("{% set _tool_calls_template = None %}")

        # Add single_observation_template if it exists
        if self.template.single_observation_template:
            header.append(
                f"{{% set _single_observation_template = {self.template.single_observation_template!r} %}}"
            )

        # Tools / skills section templates (new section-template pattern).
        # ``.format()``-escaped literal braces (``{{`` / ``}}``) are unescaped
        # because the Jinja path uses ``replace()`` rather than ``.format()``.
        if self.template.tools_template:
            tools_tpl_unescaped = self.template.tools_template.replace("{{", "{").replace("}}", "}")
            header.append(
                f"{{% set _tools_template = {tools_tpl_unescaped!r} %}}"
            )
        else:
            header.append("{% set _tools_template = None %}")
        if self.template.single_tool_template:
            single_tool_unescaped = self.template.single_tool_template.replace("{{", "{").replace("}}", "}")
            header.append(
                f"{{% set _single_tool_template = {single_tool_unescaped!r} %}}"
            )
        else:
            header.append("{% set _single_tool_template = None %}")

        # Skills section templates. When set, the generated chat template will
        # honor a ``skills`` Jinja variable (caller passes it via
        # ``tokenizer.apply_chat_template(messages, tools=..., skills=...)`` or
        # vLLM's ``chat_template_kwargs``). When the template doesn't define a
        # ``skills_template``, the ``skills`` variable is ignored.
        if self.template.skills_template:
            skills_tpl_unescaped = self.template.skills_template.replace("{{", "{").replace("}}", "}")
            header.append(
                f"{{% set _skills_template = {skills_tpl_unescaped!r} %}}"
            )
        else:
            header.append("{% set _skills_template = None %}")
        single_skill = (
            self.template.single_skill_template
            or self.template.skill_policy.single_skill_template
        )
        if single_skill:
            single_skill_unescaped = single_skill.replace("{{", "{").replace("}}", "}")
            header.append(
                f"{{% set _single_skill_template = {single_skill_unescaped!r} %}}"
            )
        else:
            header.append("{% set _single_skill_template = None %}")
        skill_joiner = self.template.skill_policy.joiner
        header.append(f"{{% set _skill_joiner = {skill_joiner!r} %}}")

        # Add generation_prompt if it exists
        if self.template.generation_prompt:
            header.append(
                f"{{% set _generation_prompt = {self.template.generation_prompt!r} %}}"
            )
        else:
            header.append("{% set _generation_prompt = None %}")

        # Add user template with tools if it exists
        if self.template.user_template_with_tools:
            # Convert double braces to single braces for Jinja compatibility
            processed_template = self.template.user_template_with_tools.replace(
                "{{", "{"
            ).replace("}}", "}")
            header.append(
                f"{{% set _user_template_with_tools = {processed_template!r} %}}"
            )

        # ------------------------------------------------------------------
        #  Formatter macro for tools (only if the template supports tool calls)
        # ------------------------------------------------------------------

        if self.template._supports_tool_call():
            # Build a Jinja macro that reproduces ToolPolicy.format_tools behaviour
            formatter_snippet = self.template.tool_policy.formatter.jinja()

            # The snippet usually comes wrapped in "{{ ... }}".  We drop the
            # outer braces because macro bodies are already an output context.
            formatter_body = formatter_snippet

            header.extend(
                [
                    "{% macro _fmt_tools(tools) %}",
                    f"{formatter_body}",
                    "{% endmacro %}",
                ]
            )

        # ------------------------------------------------------------------
        #  System processor macro (if system policy has a content processor)
        # ------------------------------------------------------------------

        if self.template.system_policy.content_processor is not None:
            # Build a Jinja macro that reproduces the system content processor behaviour
            processor_snippet = self.template.system_policy.content_processor.jinja()

            # The snippet should be a template that expects 'system_message' variable
            # We create a macro that can be called with the system message
            header.extend(
                [
                    "{% macro _process_system_message(system_message) %}",
                    f"{processor_snippet}",
                    "{% endmacro %}",
                ]
            )

        # ------------------------------------------------------------------
        #  Assistant processor macro (if assistant policy has a content processor)
        # ------------------------------------------------------------------

        if self.template.assistant_policy.content_processor is not None:
            # Build a Jinja macro that reproduces the assistant content processor behaviour
            processor_snippet = self.template.assistant_policy.content_processor.jinja()

            # The snippet should be a template that expects 'content' variable
            # We create a macro that can be called with the assistant content
            header.extend(
                [
                    "{% macro _process_assistant_content(content) %}",
                    f"{processor_snippet}",
                    "{% endmacro %}",
                ]
            )

        # ------------------------------------------------------------------
        #  Tool call processor macro (if tool policy has a tool_call_content_processor)
        # ------------------------------------------------------------------

        if self.template.tool_policy.tool_call_content_processor is not None:
            # Build a Jinja macro that reproduces the tool call content processor behaviour
            processor_snippet = (
                self.template.tool_policy.tool_call_content_processor.jinja()
            )

            # The snippet should be a template that expects 'tool' variable
            # We create a macro that can be called with the tool call
            header.extend(
                [
                    "{% macro _process_tool_call(tool) %}",
                    f"{processor_snippet}",
                    "{% endmacro %}",
                ]
            )

        return header

    def _jinja_compute_insert_idx(self) -> List[str]:
        """Return Jinja code that pre-computes the index where tools should
        be injected for FIRST_USER and LAST_USER placements."""

        return [
            "{% set _insert_ns = namespace(idx=-1) %}",
            "{% if _tool_placement in ['FIRST_USER', 'LAST_USER'] %}",
            "{%- for _m in messages -%}",
            "{%- if _m['role'] == 'user' -%}",
            "{%- if _tool_placement == 'FIRST_USER' and _insert_ns.idx == -1 -%}",
            "{% set _insert_ns.idx = loop.index0 %}",
            "{%- elif _tool_placement == 'LAST_USER' -%}",
            "{% set _insert_ns.idx = loop.index0 %}",
            "{%- endif -%}",
            "{%- endif -%}",
            "{%- endfor -%}",
            "{% endif %}",
        ]

    def _jinja_system_block(self) -> List[str]:
        """Return Jinja code that handles the system message logic.

        Section-template pattern: build a ``_tools_block`` and ``_skills_block``,
        then substitute them into ``system_template``'s ``{tools}`` / ``{skills}``
        placeholders. Each block is the empty string when its input is absent.

        Skills are passed in as a Jinja variable named ``skills`` — a list whose
        entries each expose ``name`` and ``description``. From an HF tokenizer
        call: ``tokenizer.apply_chat_template(messages, tools=..., skills=...)``.
        From vLLM's OpenAI-compat server: pass them via
        ``extra_body={"chat_template_kwargs": {"skills": ...}}``.
        """

        # Build the inner tools text + wrap with tools_template (or leave empty).
        tools_block_setup = [
            "{% if tools %}",
            "{% set _formatted_tools = _fmt_tools(tools) %}",
            "{% if _tools_template is not none %}",
            "{% set _tools_block = _tools_template | replace('{tools}', _formatted_tools) %}",
            "{% else %}",
            "{% set _tools_block = _formatted_tools %}",
            "{% endif %}",
            "{% else %}",
            "{% set _tools_block = '' %}",
            "{% endif %}",
        ]

        # Build the inner skills text + wrap with skills_template.
        # ``skills is defined`` guards against callers that omit the variable.
        skills_block_setup = [
            "{% if skills is defined and skills and _skills_template is not none and _single_skill_template is not none %}",
            "{% set _sb_ns = namespace(inner='') %}",
            "{% for skill in skills %}",
            "{% set _row = _single_skill_template | replace('{name}', skill['name']) | replace('{description}', skill['description']) %}",
            "{% if loop.first %}",
            "{% set _sb_ns.inner = _row %}",
            "{% else %}",
            "{% set _sb_ns.inner = _sb_ns.inner + _skill_joiner + _row %}",
            "{% endif %}",
            "{% endfor %}",
            "{% set _skills_block = _skills_template | replace('{skills}', _sb_ns.inner) %}",
            "{% else %}",
            "{% set _skills_block = '' %}",
            "{% endif %}",
        ]

        # Substitute system_message + the two section blocks into system_template.
        render = [
            "{% if _process_system_message is defined %}",
            "{% set _processed_system = _process_system_message(_resolved_system_message) %}",
            "{% else %}",
            "{% set _processed_system = _resolved_system_message %}",
            "{% endif %}",
            "{% set _rendered_system = _system_template | replace('{system_message}', _processed_system) | replace('{tools}', _tools_block) | replace('{skills}', _skills_block) %}",
            "{{ _rendered_system }}",
        ]

        return [
            *tools_block_setup,
            *skills_block_setup,
            "{% if messages and messages[0]['role'] == 'system' %}",
            "{% if messages[0]['content'] is string %}",
            "{% set _resolved_system_message = messages[0]['content'] %}",
            "{% else %}",
            "{% set _resolved_system_message = messages[0]['content'][0]['text'] %}",
            "{% endif %}",
            *render,
            "{% else %}",
            "{% set _resolved_system_message = _system_message %}",
            *render,
            "{% endif %}",
        ]

    def _jinja_loop_messages(self) -> List[str]:
        """Return Jinja loop that encodes all messages except the first system."""

        return [
            "{% set _tool_ns = namespace(inserted=False, user_count=0, observations=[]) %}",
            # Process remaining messages (skip first if it was system)
            "{% for m in messages %}",
            "{% if not (loop.first and m['role'] == 'system') %}",
            "{% if m['role'] == 'user' %}",
            "{% set _tool_ns.user_count = _tool_ns.user_count + 1 %}",
            "{% set ns = namespace(txt='') %}",
            "{% if m['content'] is string %}",
            "{% set ns.txt = m['content'] %}",
            "{% else %}",
            "{% for item in m['content'] %}",
            "{% if item['type'] == 'text'  %}",
            "{% set ns.txt = ns.txt + item['text'] %}",
            "{% elif item['type'] == 'image' %}",
            "{% set ns.txt = ns.txt + _image_token %}",
            "{% elif item['type'] == 'image_url' %}",
            "{% set ns.txt = ns.txt + _image_token %}",
            "{% elif item['type'] == 'video' %}",
            "{% set ns.txt = ns.txt + _video_token %}",
            "{% endif %}",
            "{% endfor %}",
            "{% endif %}",
            "{% if tools and ((_tool_placement == 'FIRST_USER' and _tool_ns.user_count == 1) or (_tool_placement == 'LAST_USER' and loop.index0 == _insert_ns.idx)) and not _tool_ns.inserted %}",
            "{% if _user_template_with_tools is defined %}",
            "{% set formatted_tools = _fmt_tools(tools) %}",
            "{{ _user_template_with_tools | replace('{content}', ns.txt) | replace('{tools}', formatted_tools) }}",
            "{% else %}",
            "{{ _user_pref }}{{ ns.txt }}{{ _user_suff }}\\n{{ _fmt_tools(tools) }}",
            "{% endif %}",
            "{% set _tool_ns.inserted = True %}",
            "{% else %}",
            "{{ _user_pref }}{{ ns.txt }}{{ _user_suff }}",
            "{% endif %}",
            "{% elif m['role'] == 'assistant' %}",
            "{% set ns = namespace(txt='', tool_calls_str='') %}",
            "{% if m['content'] is string %}",
            "{% set ns.txt = m['content'] %}",
            "{% else %}",
            "{% if m['content'] %}",
            "{% set ns.txt = m['content'][0]['text'] %}",
            "{% else %}",
            "{% set ns.txt = '' %}",
            "{% endif %}",
            "{% endif %}",
            "{% if _process_assistant_content is defined %}",
            "{% set ns.txt = _process_assistant_content(ns.txt) %}",
            "{% endif %}",
            "{% if m['tool_calls'] and _single_tool_call_template is defined %}",
            "{% for tool_call in m['tool_calls'] %}",
            "{% if _process_tool_call is defined %}",
            "{% set tool_call_str = _process_tool_call(tool_call) %}",
            "{% else %}",
            "{% set tc = tool_call %}",
            "{% if tool_call['type'] == 'function' %}",
            "{% set tc = tool_call['function'] %}",
            "{% endif %}",
            "{% set tool_call_str = tc | tojson %}",
            "{% endif %}",
            "{% set tool_call_formatted = _single_tool_call_template | replace('{tool_call}', tool_call_str) %}",
            "{% set ns.tool_calls_str = ns.tool_calls_str + tool_call_formatted %}",
            "{% endfor %}",
            "{% if _tool_calls_template is not none %}",
            "{% set ns.tool_calls_str = _tool_calls_template | replace('{tool_calls}', ns.tool_calls_str) %}",
            "{% endif %}",
            "{% endif %}",
            "{% if _supports_tool_calls %}",
            "{% set assistant_msg = _assistant_template | replace('{content}', ns.txt) | replace('{tool_calls}', ns.tool_calls_str) %}",
            "{{ assistant_msg }}",
            "{% else %}",
            "{{ _assistant_pref }}{{ ns.txt }}{{ _assistant_suff }}",
            "{% endif %}",
            "{% elif m['role'] == 'tool' %}",
            "{% if loop.first or messages[loop.index0 - 1]['role'] != 'tool' %}",
            "{% set _tool_ns.observations = [] %}",
            "{% endif %}",
            "{% set ns = namespace(txt='') %}",
            "{% if m['content'] is string %}",
            "{% set ns.txt = m['content'] %}",
            "{% else %}",
            "{% for item in m['content'] %}",
            "{% if item['type'] == 'text' %}",
            "{% set ns.txt = ns.txt + item['text'] %}",
            "{% elif item['type'] == 'image' %}",
            "{% set ns.txt = ns.txt + _image_token %}",
            "{% elif item['type'] == 'image_url' %}",
            "{% set ns.txt = ns.txt + _image_token %}",
            "{% endif %}",
            "{% endfor %}",
            "{% endif %}",
            "{% if _single_observation_template is defined %}",
            "{% set observation_formatted = _single_observation_template | replace('{observation}', ns.txt) %}",
            "{% set _tool_ns.observations = _tool_ns.observations + [observation_formatted] %}",
            "{% else %}",
            "{% set _tool_ns.observations = _tool_ns.observations + [ns.txt] %}",
            "{% endif %}",
            "{% if loop.last or (loop.index0 < messages|length - 1 and messages[loop.index0 + 1]['role'] != 'tool') %}",
            "{% set observations_combined = _tool_ns.observations | join('') %}",
            "{% if _observations_template and _uses_observations %}",
            "{{ _observations_template | replace('{observations}', observations_combined) }}",
            "{% elif _observations_template %}",
            "{{ _observations_template | replace('{observation}', observations_combined) }}",
            "{% else %}",
            "{{ _tool_pref }}{{ observations_combined }}{{ _tool_suff }}",
            "{% endif %}",
            "{% endif %}",
            "{% endif %}",
            "{% endif %}",
            "{% endfor %}",
        ]

    def _jinja_generation_block(self) -> List[str]:
        """Return Jinja code that appends the generation prefix when requested."""

        return [
            "{% if add_generation_prompt %}",
            "{% if _generation_prompt is not none %}",
            "{{ _generation_prompt }}",
            "{% else %}",
            "{{ _assistant_pref }}",
            "{% endif %}",
            "{% endif %}",
        ]
