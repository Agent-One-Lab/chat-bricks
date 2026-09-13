import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

from ..constants import Role, ToolPlacement

if TYPE_CHECKING:
    from .templates import HFTemplate, Qwen3Template, Template

logger = logging.getLogger(__name__)

# Process-wide cache of the assistant-content glue probe, keyed by template
# identity. The probe is a pure function of the template, so it runs once per
# distinct template for the life of the process -- regardless of how many
# Template copies, Renderer instances, or tokenize_conversations batches are
# created (get_template() copies base templates and rebuilds HFTemplates).
_ASSISTANT_GLUE_CACHE: Dict = {}


class Renderer:
    def __init__(self, template: "Template"):
        self.template = template

    def _render_tool_calls(self, tool_calls: List[Dict]) -> str:
        full_tool_calls_str = []
        for tool_call in tool_calls:
            if self.template.tool_policy.tool_call_content_processor is not None:
                tool_call_str = self.template.tool_policy.tool_call_content_processor(
                    tool_call
                )
            else:
                if "type" in tool_call and tool_call["type"] == "function":
                    tool_call = tool_call["function"]
                tool_call_str = json.dumps(tool_call)
            full_tool_calls_str.append(
                self.template.single_tool_call_template.format(tool_call=tool_call_str)
            )

        if self.template.tool_calls_template is not None:
            return self.template.tool_calls_template.format(
                tool_calls="".join(full_tool_calls_str)
            )
        else:
            return "".join(full_tool_calls_str)

    def _render_tool_observation(
        self, tool_observation_content: Union[str, List[Dict]]
    ) -> str:
        """
        Render parallel tool call observations into a single string. For example, for qwen3 with
        following tool observations: obs1, obs2, obs3, the rendered string should be:
        <tool_response>obs1</tool_response>\n<tool_response>obs2</tool_response>\n<tool_response>obs3</tool_response>
        """

        # If there is no single tool response template, probably model does not support
        # parallel tool calls and don't need to differentiate between single and multiple
        # tool calls, so we just return the content as is.
        if self.template.single_observation_template is None:
            if isinstance(tool_observation_content, str):
                return tool_observation_content
            elif isinstance(tool_observation_content, list):
                return "\n".join([item["text"] for item in tool_observation_content])
            else:
                raise ValueError(
                    f"Invalid tool observation content type: {type(tool_observation_content)}"
                )

        if isinstance(tool_observation_content, str):
            text = tool_observation_content
        elif isinstance(tool_observation_content, list):
            # assert len(content) == 1, "Tool message must be a single message"
            # text = content[0]["text"]
            text = ""
            for item in tool_observation_content:
                if item["type"] == "text":
                    text += item["text"]
                elif item["type"] == "image":
                    text += (
                        self.template.vision_start
                        + self.template.image_token
                        + self.template.vision_end
                    )
                # This is for openai format, since chat completion API only supports image_url
                elif item["type"] == "image_url":
                    text += (
                        self.template.vision_start
                        + self.template.image_token
                        + self.template.vision_end
                    )
                else:
                    raise ValueError(f"Invalid message type: {item['type']}")
        else:
            raise ValueError(
                f"Invalid tool observation content type: {type(tool_observation_content)}"
            )

        return self.template.single_observation_template.format(observation=text)

    def _preprocess_messages(self, messages: List[Dict]) -> List[Dict]:
        """Preprocess the messages to remove nested structures in messages
        e.g. multi (parallel) tool calls, multiple tool observations. We need to insert them into message, however, if there are multiple
        tool calls or observations, we will have to know previous or later messages to insert them correctly. This is not we want for our
        Template class.
        """
        preprocessed_messages = []
        tool_observations_str = []
        for i, message in enumerate(messages):
            if message["role"] == "assistant" and "tool_calls" in message:
                tool_calls_str = self._render_tool_calls(message["tool_calls"])
                message["tool_calls_str"] = tool_calls_str
                # Tool-call turns commonly omit ``content`` entirely (OpenAI
                # shape) or set it to ``None``; normalize both to "".
                if message.get("content") is None:
                    message["content"] = ""
                preprocessed_messages.append(message)
            elif message["role"] == "tool" and "content" in message:
                tool_observations_str.append(
                    self._render_tool_observation(message["content"])
                )
                if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                    preprocessed_messages.append(
                        {"role": "tool", "content": "".join(tool_observations_str)}
                    )
                    tool_observations_str = []
            else:
                preprocessed_messages.append(message)

        return preprocessed_messages

    def render(
        self,
        messages: List[Dict],
        tools=None,
        skills=None,
        add_generation_prompt: bool = False,
    ) -> str:
        """Render the template.

        The heavy lifting is delegated to small, single-purpose helpers so the
        high-level flow is immediately apparent:

            1. _insert_tools              – decide where the tool catalogue lives
            2. _format_skills             – format the optional skill catalogue
            3. _encode_turns              – encode every conversation turn
            4. _maybe_add_generation_prompt – append the generation prefix if requested

        Args:
            messages: The list of messages
            tools: The list of tools
            skills: Optional list of skill objects/dicts. Each entry needs ``name`` and
                ``description`` (either as dict keys or attributes). The template's
                ``skills_template`` + ``single_skill_template`` (or ``skill_policy``)
                control the rendered form. When ``skills_template`` is not set, this
                argument is silently ignored.
            add_generation_prompt: Whether to add the generation prefix

        Returns:
            prompt: The final prompt string
            elements: The list of string *elements* that compose the prompt
            mask_flags: The list of mask flags for the elements. True if the element is a non-assistant message, False if the element is an assistant message.
        """

        # Step 1 – decide tool placement & clone messages
        work_messages = self._preprocess_messages(messages)
        logger.debug(f"[Template] work_messages: {work_messages}")
        work_messages, tools_raw, tools_block, insert_tools_idx = self._insert_tools(
            work_messages, tools
        )

        # Step 1b – build the skills section (system-only; no placement variation)
        skills_str = self._format_skills(skills)

        # Step 2 – encode each conversation turn to text tokens
        elements, roles, element_token_ids = self._encode_turns(
            work_messages, tools_raw, tools_block, skills_str, insert_tools_idx
        )

        # Step 3 – append generation prefix if needed
        if add_generation_prompt:
            self._maybe_add_generation_prompt(elements, roles)
            # Any appended generation-prompt element carries no generated ids.
            element_token_ids += [None] * (len(elements) - len(element_token_ids))

        # Concatenate the prompt
        prompt = "".join(elements)

        elements, mask_flags, element_token_ids = self._postprocess_elements(
            elements, roles, element_token_ids
        )
        return prompt, elements, mask_flags, element_token_ids

    def _insert_tools(self, messages: List[Dict], tools):
        """Clone *messages* and compute where (and how) the tool catalogue
        should be injected.

        Returns:
            work_messages : List[Dict]
                A deepcopy of the original *messages* so we never mutate caller data.
            tools_raw : Optional[str]
                The raw formatted tool catalogue (no section wrapping). Used to
                fill ``{tools}`` placeholders in ``user_template_with_tools`` where
                the template author already wraps the list.
            tools_block : Optional[str]
                The system-side block: ``tools_raw`` wrapped via ``tools_template``
                if the template defines one, else identical to ``tools_raw``. Fills
                the system_template's ``{tools}`` slot.
            insert_tools_idx : int
                Index of the *user* message that receives the catalogue, or -1 when
                no injection is required.
        """

        if tools:
            tools_raw = self._format_tools_raw(tools)
            tools_block = self._wrap_tools_block(tools_raw)
            placement = self.template.tool_policy.placement
            insert_tools_idx = self._find_insert_tools_index(messages, placement)
        else:
            tools_raw = None
            tools_block = None
            insert_tools_idx = -1
        return messages, tools_raw, tools_block, insert_tools_idx

    def _format_tools_raw(self, tools) -> str:
        """Render the raw tool catalogue string (the inner list, no wrapping).

        When ``single_tool_template`` is set, wrap each tool individually and join;
        otherwise let the ``ToolPolicy.formatter`` produce the whole list in one shot.
        """
        single = self.template.single_tool_template
        if single:
            items = []
            for tool in tools:
                formatted = self.template.tool_policy.format_tools([tool])
                items.append(single.format(tool=formatted))
            return "".join(items)
        return self.template.tool_policy.format_tools(tools)

    def _wrap_tools_block(self, tools_raw: str) -> str:
        """Wrap the raw tool catalogue via ``tools_template`` for system-side placement.

        When ``tools_template`` is not set, the block is identical to the raw list.
        """
        if not self.template.tools_template:
            return tools_raw
        return self.template.tools_template.format(tools=tools_raw)

    def _format_skills(self, skills) -> str:
        """Render the skill catalogue string that fills the ``{skills}`` placeholder.

        Returns ``""`` when ``skills`` is empty or the template does not declare
        a ``skills_template`` — keeps the format() call total even for templates
        that have no skill awareness.
        """
        if not skills or not self.template.skills_template:
            return ""
        policy = self.template.skill_policy
        # Allow the template to override the policy's per-item template.
        if self.template.single_skill_template is not None:
            policy = dataclasses.replace(
                policy, single_skill_template=self.template.single_skill_template
            )
        inner = policy.format_skills(skills)
        return self.template.skills_template.format(skills=inner)

    def _encode_turns(
        self,
        work_messages: List[Dict],
        tools_raw: str,
        tools_block: str,
        skills_str: str,
        insert_tools_idx: int,
    ) -> Tuple[List[str], List[Role]]:
        """Convert every message dict into its textual representation while
        tracking roles for later masking logic.

        ``tools_block`` (wrapped) fills ``{tools}`` in the system template.
        ``tools_raw`` (unwrapped list) fills ``{tools}`` in ``user_template_with_tools``,
        whose template author handles the wrapping themselves.
        """

        elements: List[str] = []
        roles: List[Role] = []
        # Parallel to ``elements``: the model-generated token ids for an
        # assistant *content* element (splice source), or None for every other
        # element. Kept index-aligned with ``elements`` even when system/global
        # prefixes are inserted, because we append to both lists together.
        element_token_ids: List = []

        # Global prefix comes first (rarely used but must respect ordering)
        if self.template.global_policy and self.template.global_policy.prefix:
            elements.append(self.template.global_policy.prefix)
            roles.append(Role.SYSTEM)
            element_token_ids.append(None)

        for i, message in enumerate(work_messages):
            current_role = self._detect_role(message["role"])

            # --------------------------------------------------------------
            # Handle system message insertion on the very first turn
            # --------------------------------------------------------------
            if i == 0 and current_role == Role.SYSTEM:
                if self.template.system_policy.use_system:
                    system_message = self._encode_system_message(
                        message["content"], tools=tools_block, skills=skills_str
                    )
                    elements.append(system_message)
                    roles.append(Role.SYSTEM)
                    element_token_ids.append(None)
                # Whether inserted or not, we skip further handling of this
                # message because it's the (optional) system turn itself.
                continue
            elif i == 0 and current_role != Role.SYSTEM:
                if self.template.system_policy.use_system:
                    system_message = self._encode_system_message_default(
                        tools=tools_block, skills=skills_str
                    )
                    elements.append(system_message)
                    roles.append(Role.SYSTEM)
                    element_token_ids.append(None)
                # Do *not* `continue` – we still need to encode this first message.

            # --------------------------------------------------------------
            # Encode regular conversation turns
            # --------------------------------------------------------------
            if current_role == Role.USER:
                if i == insert_tools_idx:
                    user_message = self._encode_user_message_with_tools(
                        message["content"], tools=tools_raw
                    )
                else:
                    user_message = self._encode_user_message(message["content"])
                elements.append(user_message)
                roles.append(Role.USER)
                element_token_ids.append(None)

            elif current_role == Role.ASSISTANT:
                assistant_message = self._encode_assistant_message(
                    content=message["content"],
                    tool_calls_str=message["tool_calls_str"]
                    if "tool_calls_str" in message
                    else None,
                )
                elements.append(assistant_message)
                roles.append(Role.ASSISTANT)
                # Carry the generated ids (if any) so _postprocess_elements can
                # attach them to the assistant *content* sub-element after split.
                element_token_ids.append(message.get("token_ids"))

            elif current_role == Role.TOOL:
                tool_message = self._encode_tool_message(message["content"])
                elements.append(tool_message)
                roles.append(Role.TOOL)
                element_token_ids.append(None)
            else:
                raise ValueError(f"Invalid role: {message['role']}")

        return elements, roles, element_token_ids

    def _maybe_add_generation_prompt(self, elements: List[str], roles: List[Role]):
        """Append the generation prefix so the model knows to continue
        generating an assistant response."""

        generation_prefix, prefix = self._encode_generation_prompt()
        elements.append(generation_prefix)
        roles.append(Role.ASSISTANT_PREFIX)

    def _detect_role(self, role: str) -> Role:
        if role == "system":
            return Role.SYSTEM
        elif role == "user":
            return Role.USER
        elif role == "assistant":
            return Role.ASSISTANT
        elif role == "tool":
            return Role.TOOL
        else:
            raise ValueError(f"Invalid role: {role}")

    def _find_insert_tools_index(
        self, work_messages: List[Dict], placement: ToolPlacement
    ) -> int:
        insert_tools_idx = 0  # Default to insert tools at system message
        for i, message in enumerate(work_messages):
            if placement == ToolPlacement.SYSTEM:
                insert_tools_idx = 0
            elif placement == ToolPlacement.FIRST_USER:
                if message.get("role") == "user":
                    insert_tools_idx = i
                    break
            elif placement == ToolPlacement.LAST_USER:
                if message.get("role") == "user":
                    insert_tools_idx = i
            else:
                raise ValueError(f"Unhandled ToolPlacement: {placement}")
        return insert_tools_idx

    def _encode_system_message_default(self, tools=None, skills="") -> str:
        logger.debug(
            f"[Template] Encoding system message default for template: {self.template.name}"
        )
        if not self.template.system_policy.use_system_without_system_message:
            if tools is None:
                return ""
            else:
                # If tools are provided, use the system message with tools
                pass

        if self.template.system_policy.content_processor is not None:
            system_message = self.template.system_policy.content_processor(
                self.template.system_message, tools=tools
            )
        else:
            system_message = self.template.system_message

        return self._format_system_template(system_message, tools=tools, skills=skills)

    def _encode_system_message(self, content, tools=None, skills="") -> str:
        # Handle both string content and list content formats
        logger.debug(
            f"[Template] Encoding system message for template: {self.template.name}"
        )
        if isinstance(content, str):
            system_message = content
        else:
            system_message = content[0]["text"]

        if self.template.system_policy.content_processor is not None:
            system_message = self.template.system_policy.content_processor(
                system_message, tools=tools
            )

        return self._format_system_template(system_message, tools=tools, skills=skills)

    def _format_system_template(self, system_message: str, tools=None, skills: str = "") -> str:
        """Apply ``system_template``, passing ``tools=`` / ``skills=`` so templates
        that opt in to the placeholders get them filled. ``str.format()`` silently
        ignores extra kwargs, so templates without those placeholders are fine.
        """
        return self.template.system_template.format(
            system_message=system_message,
            tools=tools if tools is not None else "",
            skills=skills,
        )

    def _encode_user_message_with_tools(self, content, tools: str) -> str:
        # Handle both string content and list content formats
        if isinstance(content, str):
            text = content
        else:
            text = ""
            for item in content:
                if item["type"] == "text":
                    text += item["text"]
                elif item["type"] in ["image", "image_url"]:
                    text += (
                        self.template.vision_start
                        + self.template.image_token
                        + self.template.vision_end
                    )
                elif item["type"] == "video":
                    text += (
                        self.template.vision_start
                        + self.template.video_token
                        + self.template.vision_end
                    )
                else:
                    raise ValueError(f"Invalid message type: {item['type']}")

        if self.template.user_template_with_tools:
            user_message = self.template.user_template_with_tools.format(
                content=text, tools=tools
            )
        else:
            user_message = self.template.user_template.format(content=text)
        return user_message

    def _encode_user_message(self, content) -> str:
        # Handle both string content and list content formats
        if isinstance(content, str):
            text = content
        else:
            text = ""
            for item in content:
                if item["type"] == "text":
                    text += item["text"]
                elif item["type"] in ["image", "image_url"]:
                    text += (
                        self.template.vision_start
                        + self.template.image_token
                        + self.template.vision_end
                    )
                elif item["type"] == "video":
                    text += (
                        self.template.vision_start
                        + self.template.video_token
                        + self.template.vision_end
                    )
                else:
                    raise ValueError(f"Invalid message type: {item['type']}")
        user_message = self.template.user_template.format(content=text)
        return user_message

    def _encode_assistant_message(self, content, tool_calls_str=None) -> str:
        if isinstance(content, str):
            text = content
        else:
            assert len(content) == 1, "Assistant message must be a single message"
            text = content[0]["text"]

        if self.template.assistant_policy.content_processor is not None:
            text = self.template.assistant_policy.content_processor(text)

        if "{tool_calls}" in self.template.assistant_template:
            assistant_message = self.template.assistant_template.format(
                content=text, tool_calls=tool_calls_str if tool_calls_str else ""
            )
        else:
            assistant_message = self.template.assistant_template.format(content=text)

        logger.debug(
            f"[Template] tool_calls_str: {tool_calls_str}, assistant_message: {assistant_message}"
        )

        return assistant_message

    def _encode_tool_message(self, content) -> str:
        """
        Encode the tool message. By default, we use "observations" as placeholder for parallel tool calls and "observation" for single tool call.
        """
        # We have already preprocessed the messages, so content should be a string
        assert isinstance(content, str), (
            f"Content should be a string, but got {type(content)}"
        )

        if "{observations}" in self.template.observations_template:
            tool_message = self.template.observations_template.format(observations=content)
        else:
            tool_message = self.template.observations_template.format(observation=content)
        return tool_message

    def _encode_generation_prompt(self) -> str:
        # Use generation prompt if it is set
        if "{content}" in self.template.assistant_template:
            prefix = self.template.assistant_template.split("{content}")[0]
            if self.template.generation_prompt:
                generation_prompt = self.template.generation_prompt
            else:
                generation_prompt = prefix
        else:
            raise ValueError(
                f"Assistant template {self.template.assistant_template} does not contain {{content}}"
            )

        return generation_prompt, prefix

    def _split_assistant_message(self, assistant_message: str) -> List[str]:
        # Split the assistant message into generation prefix, content, and generation suffix
        generation_prefix, prefix = self._encode_generation_prompt()
        assert assistant_message.startswith(prefix), (
            f"Assistant message {assistant_message} does not start with {prefix}"
        )
        content_suffix = assistant_message[len(prefix) :]
        content = content_suffix
        suffix = ""
        for stop_word in self.template.stop_words:
            if stop_word in content_suffix:
                stop_word_index = content_suffix.index(stop_word)
                content = content_suffix[: stop_word_index + len(stop_word)]
                suffix = content_suffix[stop_word_index + len(stop_word) :]
                break
        return prefix, content, suffix

    def _assistant_glue(self):
        """The template's structural glue around assistant-generated content,
        as ``(leading, trailing)`` strings, discovered by a one-time sentinel
        probe -- never from any generated ``token_ids``.

        ``leading`` is text the template places *before* the model output inside
        the trained content span (e.g. the base Qwen template fuses a role
        separator ``"\\n"`` there); ``trailing`` is the content span's tail
        *starting at the turn terminator* (e.g. ``"<|im_end|>"``). The encoder
        prepends ``leading``, uses the generated ids verbatim, and appends
        ``trailing`` minus its leading terminator (matched by id). Cached on the
        template so the (pure string) probe runs once.
        """
        # get_template() hands out a fresh copy() of a base template per call, so
        # an instance attribute would be re-probed every conversation. Key a
        # module-level cache by the fields that determine the glue instead, so
        # every copy (and every render) shares one probe.
        key = (
            self.template.name,
            self.template.assistant_template,
            tuple(self.template.stop_words or ()),
        )
        cached = _ASSISTANT_GLUE_CACHE.get(key)
        if cached is not None:
            return cached
        sentinel = "chatbricks_content_sentinel"
        leading, trailing = "", ""
        try:
            asst = self._encode_assistant_message(sentinel)
            _prefix, content, _suffix = self._split_assistant_message(asst)
            idx = content.find(sentinel)
            if idx != -1:
                leading = content[:idx]
                trailing = content[idx + len(sentinel):]
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "[chat-bricks] assistant-glue probe failed for %s (%s); "
                "token_ids splice will add no surrounding glue.",
                self.template.name,
                e,
            )
        _ASSISTANT_GLUE_CACHE[key] = (leading, trailing)
        return leading, trailing

    def _spliced_prefix(self, split_prefix: str) -> str:
        """The masked text that precedes spliced ``token_ids`` for an assistant turn.

        Sampled ids are, by definition, the model's continuation of the generation
        prompt it was given, so the prefix of a spliced turn is that generation
        prompt -- not the assistant-template prefix, which some templates define
        differently (e.g. the base Qwen template fuses the role separator into the
        content slot). Renderers whose ``_split_assistant_message`` already returns a
        generation-aware prefix override this to keep it.
        """
        generation_prompt, _prefix = self._encode_generation_prompt()
        return generation_prompt

    def _postprocess_elements(
        self, elements: List[str], roles, element_token_ids=None
    ) -> List[str]:
        # ``element_token_ids`` is parallel to ``elements``: the generated ids
        # for an assistant turn (splice source) or None. When absent, behave
        # exactly as before (all None → pure text encoding downstream).
        if element_token_ids is None:
            element_token_ids = [None] * len(elements)

        # Flag non-assistant messages
        new_elements = []
        mask_flags = []
        for i, element in enumerate(elements):
            if roles[i] == Role.ASSISTANT:
                new_elements.append(element)
                mask_flags.append(False)
            else:
                new_elements.append(element)
                mask_flags.append(True)

        # return new_elements, mask_flags

        # merge non-assistant messages and handle the generation prefix and suffixes
        merged_elements = []
        merged_mask_flags = []
        # Parallel to ``merged_elements``: only an assistant *content* sub-element
        # carries the turn's generated ids; every prefix/suffix/merged-context
        # element carries None (deterministic template glue, encoded as text).
        merged_element_token_ids = []

        for i, (element, mask_flag) in enumerate(zip(new_elements, mask_flags)):
            if i == 0:
                prev_element = element
                prev_mask_flag = mask_flag
                continue
            else:
                if prev_mask_flag == mask_flag:
                    # Both previous and current elements are assistant messages
                    if not mask_flag:
                        prefix, content, suffix = self._split_assistant_message(element)
                        _ids = element_token_ids[i]
                        if _ids is not None:
                            # Spliced turn: the sampled ids continue the generation
                            # prompt the model was given, so that prompt is the masked
                            # prefix and nothing sits between it and the ids.
                            prefix = self._spliced_prefix(prefix)
                        merged_elements.append(prefix)
                        merged_mask_flags.append(True)
                        merged_element_token_ids.append(None)
                        merged_elements.append(content)
                        merged_mask_flags.append(False)
                        _lead, _trail = self._assistant_glue()
                        merged_element_token_ids.append(
                            (_ids, "", _trail) if _ids is not None else None
                        )
                        prev_element = suffix
                        prev_mask_flag = True  # We need to mask the suffix
                    # Both previous and current elements are non-assistant messages
                    else:
                        prev_element += element
                        prev_mask_flag = True
                else:
                    # Previous element is not assistant message, but the current one is
                    if not mask_flag:
                        prefix, content, suffix = self._split_assistant_message(element)
                        _ids = element_token_ids[i]
                        if _ids is not None:
                            prefix = self._spliced_prefix(prefix)  # see above
                        prev_element += prefix
                        prev_mask_flag = True
                        merged_elements.append(prev_element)
                        merged_mask_flags.append(prev_mask_flag)
                        merged_element_token_ids.append(None)
                        merged_elements.append(content)
                        merged_mask_flags.append(False)
                        _lead, _trail = self._assistant_glue()
                        merged_element_token_ids.append(
                            (_ids, "", _trail) if _ids is not None else None
                        )
                        prev_element = suffix
                        prev_mask_flag = True
                    # Previous element is assistant message, but the current one is not
                    else:
                        prev_element += element
                        prev_mask_flag = True
        if prev_element != "":
            merged_elements.append(prev_element)
            merged_mask_flags.append(prev_mask_flag)
            merged_element_token_ids.append(None)
        return merged_elements, merged_mask_flags, merged_element_token_ids


class Qwen3Renderer(Renderer):
    def __init__(self, template: "Qwen3Template"):
        super().__init__(template)

    def _encode_assistant_message(self, content, tool_calls_str=None) -> str:
        if isinstance(content, str):
            text = content
        else:
            assert len(content) == 1, "Assistant message must be a single message"
            text = content[0]["text"]
        return super()._encode_assistant_message(text, tool_calls_str)

    def render(
        self,
        messages: List[Dict],
        tools=None,
        skills=None,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        """Render the Qwen3 template with special thinking logic.

        Args:
            messages: The list of messages
            tools: The list of tools
            skills: Optional list of skill objects/dicts.
            add_generation_prompt: Whether to add the generation prefix
            enable_thinking: Whether to enable thinking mode

        Returns:
            prompt: The final prompt string
            elements: The list of string *elements* that compose the prompt
            roles: The corresponding list of *roles* (used by downstream post-processing)
        """

        # Step 1 – decide tool placement & clone messages
        work_messages = self._preprocess_messages(messages)
        logger.debug(f"[Qwen3Template] work_messages: {work_messages}")
        work_messages, tools_raw, tools_block, insert_tools_idx = self._insert_tools(
            work_messages, tools
        )
        skills_str = self._format_skills(skills)

        # Step 2 – clean think content from all assistant messages except the last one
        work_messages = self._clean_think_content(work_messages)

        # Step 2.5 – reformat think content in the last assistant message if it exists
        if work_messages and work_messages[-1].get("role") == "assistant":
            work_messages = self._reformat_last_assistant_think_content(work_messages)

        # Step 3 – encode each conversation turn to text tokens
        elements, roles, element_token_ids = self._encode_turns(work_messages, tools_raw, tools_block, skills_str, insert_tools_idx)

        # Step 4 – handle special generation prompt logic for Qwen3
        if add_generation_prompt:
            self._maybe_add_generation_prompt_qwen3(
                elements, roles, enable_thinking, work_messages
            )
        elif work_messages and work_messages[-1].get("role") == "assistant":
            # Add empty think tokens to the last assistant message if it doesn't already have think tags
            self._add_empty_think_to_last_assistant(elements, roles, work_messages)
        # The helpers above may append/rewrite elements; keep the parallel
        # token-ids list aligned (appended glue carries no generated ids).
        element_token_ids += [None] * (len(elements) - len(element_token_ids))

        # Concatenate the prompt
        prompt = "".join(elements)
        elements, mask_flags, element_token_ids = self._postprocess_elements(
            elements, roles, element_token_ids
        )
        return prompt, elements, mask_flags, element_token_ids

    def _clean_think_content(self, messages: List[Dict]) -> List[Dict]:
        """Remove all think content (<think>...</think>) from assistant messages and reformat existing think content."""
        cleaned_messages = []
        for i, message in enumerate(messages):
            if message.get("role") == "assistant" and i != len(messages) - 1:
                cleaned_message = message.copy()
                content = message["content"]

                if isinstance(content, str):
                    # Remove think content from string
                    cleaned_content = self._remove_think_tags(content)
                elif isinstance(content, list):
                    # Handle list content format
                    cleaned_content = []
                    for item in content:
                        if item["type"] == "text":
                            cleaned_text = self._remove_think_tags(item["text"])
                            cleaned_content.append(
                                {"type": "text", "text": cleaned_text}
                            )
                        else:
                            cleaned_content.append(item)
                elif content is None:
                    cleaned_content = ""
                else:
                    raise ValueError(f"Invalid content type: {type(content)}")

                cleaned_message["content"] = cleaned_content
                # This turn's content was rewritten (think stripped), so the
                # generated token_ids no longer correspond to what will be
                # rendered. Drop them: the turn must be text-encoded from the
                # rewritten content, not spliced verbatim.
                cleaned_message.pop("token_ids", None)
                cleaned_messages.append(cleaned_message)
            else:
                cleaned_messages.append(message)

        return cleaned_messages

    def _remove_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from text."""
        import re

        # Remove <think>...</think> tags and their content
        pattern = r"<think>.*?</think>"
        return re.sub(pattern, "", text, flags=re.DOTALL)

    def _has_think_tags(self, text: str) -> bool:
        """Check if text contains <think> and </think> tags."""
        return "<think>" in text and "</think>" in text

    def _reformat_think_content(self, text: str) -> str:
        """Reformat think content to ensure each think token ends with two newlines."""
        import re

        def replace_think_content(match):
            think_content = match.group(1)
            # Ensure the think content ends with exactly two newlines
            think_content = think_content.rstrip("\n")
            return f"<think>\n{think_content}\n</think>\n\n"

        # Find and replace think tags, ensuring proper formatting
        pattern = r"<think>(.*?)</think>"
        return re.sub(pattern, replace_think_content, text, flags=re.DOTALL)

    def _reformat_last_assistant_think_content(
        self, messages: List[Dict]
    ) -> List[Dict]:
        """Reformat think content in the last assistant message."""
        if not messages or messages[-1].get("role") != "assistant":
            return messages

        messages = messages.copy()
        last_message = messages[-1].copy()
        content = last_message["content"]

        if isinstance(content, str):
            # Reformat think content in string
            last_message["content"] = self._reformat_think_content(content)
        else:
            # Handle list content format
            reformed_content = []
            for item in content:
                if item["type"] == "text":
                    reformed_text = self._reformat_think_content(item["text"])
                    reformed_content.append({"type": "text", "text": reformed_text})
                else:
                    reformed_content.append(item)
            last_message["content"] = reformed_content

        messages[-1] = last_message
        return messages

    def _maybe_add_generation_prompt_qwen3(
        self,
        elements: List[str],
        roles: List[Role],
        enable_thinking: bool,
        work_messages: List[Dict],
    ):
        """Append the generation prefix with special Qwen3 thinking logic."""
        if enable_thinking:
            # Use standard generation prompt
            generation_prefix, prefix = self._encode_generation_prompt()
            elements.append(generation_prefix)
            roles.append(Role.ASSISTANT_PREFIX)
        else:
            # Check if the last message has think tags
            has_existing_think = False
            if work_messages and work_messages[-1].get("role") == "assistant":
                content = work_messages[-1]["content"]
                if isinstance(content, str):
                    has_existing_think = self._has_think_tags(content)
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text" and self._has_think_tags(
                            item["text"]
                        ):
                            has_existing_think = True
                            break

            generation_prefix, prefix = self._encode_generation_prompt()
            if has_existing_think:
                # Don't add empty think tokens if think tags already exist
                elements.append(generation_prefix)
            else:
                # Add empty think tokens after the generation prefix
                elements.append(generation_prefix + "<think>\n\n</think>\n\n")
            roles.append(Role.ASSISTANT_PREFIX)

    def _add_empty_think_to_last_assistant(
        self, elements: List[str], roles: List[Role], work_messages: List[Dict]
    ):
        """Add empty think tokens to the last assistant message if it doesn't already have think tags."""
        if not elements or not roles or not work_messages:
            return

        # Check if the last message has think tags
        has_existing_think = False
        if work_messages[-1].get("role") == "assistant":
            content = work_messages[-1]["content"]
            if isinstance(content, str):
                has_existing_think = self._has_think_tags(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text" and self._has_think_tags(
                        item["text"]
                    ):
                        has_existing_think = True
                        break

        # Only add empty think tokens if no existing think tags
        if not has_existing_think:
            generation_prefix, prefix = self._encode_generation_prompt()

            # Find the last assistant element
            for i in range(len(elements) - 1, -1, -1):
                if roles[i] == Role.ASSISTANT:
                    # Add empty think tokens at the start of the assistant message
                    elements[i] = (
                        generation_prefix
                        + "<think>\n\n</think>\n\n"
                        + elements[i][len(generation_prefix) :]
                    )
                    break

    def _spliced_prefix(self, split_prefix: str) -> str:
        # ``_split_assistant_message`` below already returns the generation prefix
        # (plus the empty-think block for a thinking-off turn), i.e. exactly what the
        # model was given before sampling.
        return split_prefix

    def _split_assistant_message(self, assistant_message: str) -> List[str]:
        # Split the assistant message into generation prefix, content, and generation suffix
        generation_prefix, prefix = self._encode_generation_prompt()
        assert assistant_message.startswith(prefix), (
            f"Assistant message {assistant_message} does not start with {prefix}"
        )

        # We need to detect whether the assistant message starts with empty think tokens
        # If so, we need to set empty think tokens as non-assistant message
        if assistant_message.startswith(generation_prefix + "<think>\n\n</think>\n\n"):
            generation_prefix = generation_prefix + "<think>\n\n</think>\n\n"

        content_suffix = assistant_message[len(generation_prefix) :]
        content = content_suffix
        suffix = ""
        for stop_word in self.template.stop_words:
            if stop_word in content_suffix:
                stop_word_index = content_suffix.index(stop_word)
                content = content_suffix[: stop_word_index + len(stop_word)]
                suffix = content_suffix[stop_word_index + len(stop_word) :]
                break
        return generation_prefix, content, suffix


class HFRenderer(Renderer):
    def __init__(self, template: "HFTemplate"):
        super().__init__(template)
        self._assistant_glue_cache = None
        self._warned_history_rewrite = False

    def _assistant_glue(self):
        """``(leading, trailing)`` structural glue around assistant content for
        this HF chat template, discovered by a one-time sentinel probe.

        Render a throwaway assistant turn with a sentinel body, isolate the
        assistant content element (the diff after the generation prompt, using
        the same prefix-diff HFRenderer uses elsewhere), and read off the text
        before the sentinel (leading) and from the sentinel onward (trailing,
        which starts at the turn terminator). Never references any generated
        token_ids; hardcodes no terminator string. Cached (one per HFTemplate).
        """
        if self._assistant_glue_cache is not None:
            return self._assistant_glue_cache
        # get_template() rebuilds an HFTemplate (and this renderer) per batch;
        # the module-level cache keyed by model name keeps the probe to once
        # per process instead of once per batch.
        key = ("hf", self.template.name)
        cached = _ASSISTANT_GLUE_CACHE.get(key)
        if cached is not None:
            self._assistant_glue_cache = cached
            return cached
        sentinel = "chatbricks_content_sentinel"
        leading, trailing = "", ""
        try:
            probe = [
                {"role": "user", "content": "probe"},
                {"role": "assistant", "content": sentinel},
            ]
            with_asst = self.template.tokenizer.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=False
            )
            gen = self.template.tokenizer.apply_chat_template(
                probe[:1], tokenize=False, add_generation_prompt=True
            )
            content_elem = with_asst[self._find_prefix_length(gen, with_asst):]
            idx = content_elem.find(sentinel)
            if idx != -1:
                leading = content_elem[:idx]
                trailing = content_elem[idx + len(sentinel):]
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "[chat-bricks] assistant-glue probe failed for %s (%s); "
                "token_ids splice will add no surrounding glue.",
                self.template.name,
                e,
            )
        self._assistant_glue_cache = (leading, trailing)
        _ASSISTANT_GLUE_CACHE[key] = (leading, trailing)
        return leading, trailing


    def _find_prefix_length(self, text1: str, text2: str) -> int:
        """Find the length of the longest common prefix between two strings."""
        min_len = min(len(text1), len(text2))
        for i in range(min_len):
            if text1[i] != text2[i]:
                return i
        return min_len

    def _render_message_turn(
        self,
        messages: List[Dict],
        turn_idx: int,
        prev_prompt: str,
        tools: List[Dict] = None,
        **kwargs,
    ) -> Tuple[str, str, bool]:
        """Render a single message turn and extract new content.

        Args:
            messages: All messages
            turn_idx: Index of current turn to render
            prev_prompt: Previous prompt (may include gen prompt if prev_has_gen_prompt is True)
            prev_has_gen_prompt: Whether prev_prompt ends with a generation prompt
            tools: Tools to include (only used for first turn)
            **kwargs: Additional kwargs for apply_chat_template

        Returns:
            current_prompt: The rendered prompt up to this turn (without gen prompt)
            new_element: The new content added in this turn (empty string if none)
            is_assistant: Whether this turn is an assistant message
        """
        current_messages = messages[: turn_idx + 1]
        current_tools = tools

        # Always render without gen prompt to get clean content
        # When prev_has_gen_prompt is True, prev_prompt includes the gen prompt (assistant prefix),
        # so comparing prev_prompt to current_prompt will correctly extract only the content
        current_prompt = self.template.tokenizer.apply_chat_template(
            current_messages,
            tokenize=False,
            tools=current_tools,
            add_generation_prompt=False,
            **kwargs,
        )

        # Extract new content: difference between prev_prompt and current_prompt
        # If prev_has_gen_prompt, prev_prompt ends with gen prompt, so we get content after it
        # Otherwise, we get the full new turn content
        prefix_len = self._find_prefix_length(prev_prompt, current_prompt)
        is_assistant = messages[turn_idx].get("role") == "assistant"
        if prefix_len < len(prev_prompt) and not is_assistant:
            # The template rewrote history that was already emitted (Qwen3-style
            # templates drop the reasoning of earlier assistant turns once a later
            # user query appears). A plain diff would then re-emit part of the
            # earlier turn inside this element. Isolate this turn's own text
            # instead; the previously emitted elements stay as they were (spliced
            # ids are unaffected; text-path assistant turns keep the reasoning the
            # model actually produced).
            new_element = self._render_turn_delta_by_repeat(
                messages, turn_idx, current_prompt, tools, **kwargs
            )
            if not self._warned_history_rewrite:
                self._warned_history_rewrite = True
                logger.warning(
                    "[chat-bricks] template %s rewrote earlier turns when message %d "
                    "(%s) was appended; rendering that turn by isolation. Earlier "
                    "assistant elements keep their originally rendered text.",
                    self.template.name, turn_idx, messages[turn_idx].get("role"),
                )
        else:
            new_element = current_prompt[prefix_len:]
        return current_prompt, new_element, is_assistant

    def _render_turn_delta_by_repeat(
        self, messages, turn_idx, current_prompt, tools=None, **kwargs
    ) -> str:
        """This turn's rendered text, isolated by rendering the turn twice.

        With the same message appended once more, both renders share the same
        history (the duplicate is what moves any "last query" bookkeeping), so the
        second copy's delta is exactly the turn's own rendering, which must also be
        the tail of ``current_prompt``.
        """
        once = messages[: turn_idx + 1]
        twice = once + [messages[turn_idx]]
        rendered_twice = self.template.tokenizer.apply_chat_template(
            twice, tokenize=False, tools=tools, add_generation_prompt=False, **kwargs
        )
        role = messages[turn_idx].get("role")
        if not rendered_twice.startswith(current_prompt):
            raise ValueError(
                f"[chat-bricks] cannot isolate turn {turn_idx} ({role}) for template "
                f"{self.template.name}: repeating the message changed the rendering of "
                "earlier turns."
            )
        delta = rendered_twice[len(current_prompt):]
        if not delta or not current_prompt.endswith(delta):
            raise ValueError(
                f"[chat-bricks] cannot isolate turn {turn_idx} ({role}) for template "
                f"{self.template.name}: the repeated message does not render as a suffix "
                "of the conversation."
            )
        return delta

    def _add_generation_prompt(
        self,
        messages: List[Dict],
        turn_idx: int,
        current_prompt: str,
        tools=None,
        **kwargs,
    ) -> Tuple[str, str]:
        """Add generation prompt after a turn if the next turn is assistant.

        Returns:
            gen_prompt: The prompt with generation prompt added
            gen_element: The generation prompt element to add (empty string if none)
        """
        current_messages = messages[: turn_idx + 1]
        current_tools = tools

        gen_prompt = self.template.tokenizer.apply_chat_template(
            current_messages,
            tokenize=False,
            tools=current_tools,
            add_generation_prompt=True,
            **kwargs,
        )
        prefix_len = self._find_prefix_length(current_prompt, gen_prompt)
        gen_element = gen_prompt[prefix_len:]
        return gen_prompt, gen_element

    def render(
        self,
        messages: List[Dict],
        tools=None,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Tuple[str, List[str], List[bool]]:
        """
        Render the HF template by progressively rendering messages to identify
        elements and their mask flags.

        Only renders at key boundaries:
        - Before assistant turns (to get generation prompts)
        - At assistant turns (to get assistant content)
        This handles templates that combine successive messages (e.g., tool messages)
        and reduces computation.

        Returns:
            prompt: The final prompt string
            elements: The list of string *elements* that compose the prompt
            mask_flags: The list of mask flags for the elements. True if the element is a non-assistant message, False if the element is an assistant message.
        """
        if not messages:
            prompt = self.template.tokenizer.apply_chat_template(
                [],
                tokenize=False,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            return (
                prompt,
                [prompt],
                [True],
                [None],
            )  # Generation prompt or empty prompt is masked

        elements: List[str] = []
        mask_flags: List[bool] = []
        # Parallel to ``elements``: generated ids for an assistant content
        # element (splice source), None for every non-assistant / glue element.
        element_token_ids: List = []
        prev_prompt = ""
        # prev_has_gen_prompt = False

        # Only render at key boundaries:
        # 1. Before assistant turns (to get generation prompts)
        # 2. At assistant turns (to get assistant content)
        # 3. At the last turn (to capture content after last assistant, e.g., tool messages)
        # This handles templates that combine successive messages (e.g., tool messages)
        assistant_indices = [
            i for i, msg in enumerate(messages) if msg.get("role") == "assistant"
        ]
        render_indices = set()

        # Add indices: assistant turns and turns before assistant turns
        for idx in assistant_indices:
            render_indices.add(idx)  # Assistant turn itself
            if idx > 0:
                render_indices.add(idx - 1)  # Turn before assistant

        # Add last turn if not already covered (to capture tool messages after last assistant)
        if messages and len(messages) - 1 not in render_indices:
            render_indices.add(len(messages) - 1)

        # Process only at key boundaries (in order)
        for i in sorted(render_indices):
            # is_assistant = messages[i].get("role") == "assistant"
            next_is_assistant = (
                i + 1 < len(messages) and messages[i + 1].get("role") == "assistant"
            )

            # Render this turn and extract new content
            current_prompt, new_element, turn_is_assistant = self._render_message_turn(
                messages, i, prev_prompt, tools, **kwargs
            )

            # Add new content element
            if new_element:
                elements.append(new_element)
                mask_flags.append(
                    not turn_is_assistant
                )  # True for non-assistant, False for assistant
                # Attach the generated ids only to an assistant content element,
                # paired with the template's structural glue (leading text +
                # trailing tail from the terminator) so the encoder can stitch by
                # token id without inspecting the (possibly re-serialized) text.
                _ids = messages[i].get("token_ids") if turn_is_assistant else None
                if _ids is not None:
                    # The generation prompt was emitted as its own masked element right
                    # before this turn, and the sampled ids continue it verbatim, so no
                    # leading glue may be inserted. (The sentinel probe's ``leading`` is
                    # mode-dependent -- for a thinking template it is the empty-think
                    # closer -- and was wrong for a thinking-on sample.) Only the
                    # post-terminator glue is taken from the probe.
                    _lead, _trail = self._assistant_glue()
                    element_token_ids.append((_ids, "", _trail))
                else:
                    element_token_ids.append(None)

            # If next turn is assistant, add generation prompt
            if next_is_assistant:
                gen_prompt, gen_element = self._add_generation_prompt(
                    messages, i, current_prompt, tools, **kwargs
                )
                if gen_element:
                    elements.append(gen_element)
                    mask_flags.append(True)  # Generation prompt is masked
                    element_token_ids.append(None)
                prev_prompt = gen_prompt
                # prev_has_gen_prompt = True
            else:
                prev_prompt = current_prompt
                # prev_has_gen_prompt = False

        # Handle final generation prompt for inference
        if add_generation_prompt:
            gen_prompt = self.template.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                tools=tools,
                add_generation_prompt=True,
                **kwargs,
            )
            prefix_len = self._find_prefix_length(prev_prompt, gen_prompt)
            gen_element = gen_prompt[prefix_len:]
            if gen_element:
                elements.append(gen_element)
                mask_flags.append(True)
                element_token_ids.append(None)
            final_prompt = gen_prompt
        else:
            # Render final prompt with all messages and tools to ensure tools are included
            final_prompt = self.template.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                tools=tools,
                add_generation_prompt=False,
                **kwargs,
            )

        return final_prompt, elements, mask_flags, element_token_ids
