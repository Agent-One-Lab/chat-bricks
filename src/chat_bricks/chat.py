import logging
from typing import Any, Dict, List, Union

from transformers import PreTrainedTokenizer

from .registry import get_template
from .templates import HFTemplate, Template
from .utils import is_vision_lm

logger = logging.getLogger(__name__)


class Chat:
    def __init__(
        self,
        template: str | Template | HFTemplate,
        messages: List[List[str]] = None,
        tools=None,
        skills=None,
        tokenizer: PreTrainedTokenizer = None,
        ignore_tool_calls: bool = False,
    ):
        """
        Args:
            template: The name of the template to use.
            messages: The messages to use for the chat.
            tools: The tools to use for the chat.
            skills: Optional list of skill objects/dicts to advertise via the template's
                ``{skills}`` placeholder (requires ``skills_template`` on the template).
            tokenizer: The tokenizer to use for the chat.
        """
        if isinstance(template, str):
            self.template = get_template(template, tokenizer=tokenizer)
        else:
            self.template = template

        # Default to use vision template because we use vision format messages by default.
        # For HF template, detect model capability to decide whether to keep vision-format
        # content blocks or convert text-only list blocks to plain strings.
        if isinstance(self.template, HFTemplate):
            self.is_vision_template = is_vision_lm(self.template.name)
        else:
            self.is_vision_template = True

        self.ignore_tool_calls = ignore_tool_calls
        self.messages = self.preprocess_messages(messages)
        logger.debug(f"[chat-bricks/Chat] Messages: {self.messages}")
        self.tokenizer = tokenizer
        self.tools = tools
        self.skills = skills
        self.flags = {}

    def _detect_labels(self, messages):
        message = messages[0]
        if "role" in message and "content" in message:
            return "role", "content"
        elif "from" in message and "value" in message:
            return "from", "value"
        else:
            raise ValueError("Cannot find role label and content label in the data.")

    def preprocess_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Preprocess the messages for the chat. Since we don't change message content, we don't copy it here.
        """
        processed_messages = []

        for message in messages:
            processed_message = {}
            for k, v in message.items():
                if k == "tool_calls" and self.ignore_tool_calls:
                    continue
                processed_message[k] = v
            processed_messages.append(processed_message)

        processed_messages = self.convert_to_hf_format_messages(processed_messages)

        return processed_messages

    def _convert_single_message_to_hf_format(self, message: Dict) -> Dict:
        if isinstance(message["content"], str):
            # Convert to vision format is we use defined template or HF's vision template.
            if self.is_vision_template:
                message["content"] = [{"type": "text", "text": message["content"]}]
        elif isinstance(message["content"], list):
            if self.is_vision_template:
                for item in message["content"]:
                    if item["type"] == "text":
                        continue
                    elif item["type"] in ["image", "image_url"]:
                        pass
                    else:
                        raise ValueError(f"Invalid message type: {item['type']}")
            else:
                if (
                    len(message["content"]) == 1
                    and message["content"][0]["type"] == "text"
                ):
                    message["content"] = message["content"][0]["text"]

    def convert_to_hf_format_messages(
        self, messages: Union[List[Dict], Dict[str, List[Dict]]]
    ) -> List[Dict]:
        hf_messages = []

        if messages is None:
            return None

        role_label, content_label = self._detect_labels(messages)
        for message in messages:
            hf_message = {
                "role": message[role_label],
                "content": message[content_label],
            }
            if "tool_calls" in message:
                hf_message["tool_calls"] = message["tool_calls"]
            hf_messages.append(hf_message)

        for message in hf_messages:
            self._convert_single_message_to_hf_format(message)

        return hf_messages

    def set_messages(self, messages: List[Dict]):
        """Set the messages for the chat."""
        self.messages = self.convert_to_hf_format_messages(messages)

    def prompt(self, add_generation_prompt=False, tools=None, skills=None, **kwargs) -> str:
        """Get the prompt for the chat.

        Args:
            add_generation_prompt: Whether to add the generation prompt.
            tools: The tools to use for the chat.
            skills: Optional list of skill objects/dicts (see ``Chat.__init__``).
            **kwargs: Additional keyword arguments to pass to the template render method.

        Returns:
            The string formatted prompt for the messages after applying the chat template.
        """
        self.flags["add_generation_prompt"] = add_generation_prompt
        tools = tools or self.tools
        skills = skills if skills is not None else self.skills
        prompt, _, _ = self.template.render(
            messages=self.messages,
            tools=tools,
            skills=skills,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        return prompt

    def prompt_with_mask(
        self, add_generation_prompt=False, tools=None, skills=None, **kwargs
    ) -> str:
        """Get the prompt for the chat with highlight on the masked parts.

        Args:
            add_generation_prompt: Whether to add the generation prompt.
            tools: The tools to use for the chat.
            skills: Optional list of skill objects/dicts (see ``Chat.__init__``).
            **kwargs: Additional keyword arguments to pass to the template render method.

        Returns:
            The string formatted prompt for the messages after applying the chat template with highlight on the masked parts.
        """
        tools = tools or self.tools
        skills = skills if skills is not None else self.skills
        prompt_with_mask, _, _ = self.template.render_with_mask(
            messages=self.messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            skills=skills,
            **kwargs,
        )
        return prompt_with_mask

    def vision_inputs(self) -> List[Any]:
        return self.template.get_vision_inputs(self.messages)

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizer = None,
        add_generation_prompt=False,
        tools=None,
        skills=None,
        processor=None,
        train_on_last_turn_only=False,
        **kwargs,
    ) -> List[int]:
        """Tokenize the messages.

        Args:
            tokenizer: The tokenizer to use for the chat.
            add_generation_prompt: Whether to add the generation prompt.
            tools: The tools to use for the chat.
            skills: Optional list of skill objects/dicts (see ``Chat.__init__``).
            processor: The processor to use for the chat.

        Returns:
            inputs (dict): Inputs for helping training.
                - input_ids
                - attention_mask
                - labels
                - action_mask
                - multi_modal_inputs
        """
        if tokenizer is None:
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is not set. Set it when initializing the chat or pass it as an argument."
                )
            tokenizer = self.tokenizer

        if tools is None:
            tools = self.tools
        if skills is None:
            skills = self.skills

        return self.template.encode(
            messages=self.messages,
            tokenizer=tokenizer,
            return_tensors="pt",
            tools=tools,
            skills=skills,
            add_generation_prompt=add_generation_prompt,
            processor=processor,
            train_on_last_turn_only=train_on_last_turn_only,
            **kwargs,
        )

    def append(self, message: Union[Dict]):
        self._convert_single_message_to_hf_format(message)
        self.messages.append(message)
