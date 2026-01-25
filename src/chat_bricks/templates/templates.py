
from collections import defaultdict
from copy import copy, deepcopy
import dataclasses
import json
from typing import Callable, List, Any, Dict, Union, Tuple
import warnings
import logging
import torch
from transformers import PreTrainedTokenizer
import re
from typing import Protocol
from ..policies import (
    ToolFormatter,
    JsonMinifiedFormatter,
    JsonCompactFormatter,
    JsonIndentedFormatter,
    ToolMainContentProcessor,
    JsonQwenFormatter,
)
from .renderer import Renderer, Qwen3Renderer
from .jinja_generator import JinjaGenerator
from datetime import datetime
from ..constants import Role
from ..policies import Llama32DateProcessor, SystemPolicy
from ..policies import AssistantPolicy, Qwen25AssistantContentProcessor
from ..policies import ToolPolicy
from ..constants import ToolPlacement, Role
from ..policies import GlobalPolicy
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



@dataclasses.dataclass
class Template:
    """Class that holds all the components of a chat template. Convert messages to string prompts, tokenize messages to token ids, and generate jinja-based chat templates.

    Args:
        name: The name of this template
        system_template: The system template component
        system_template_with_tools: The system template with tool usage component
        system_message: The default system message
        stop_words: The stop words where the model stops generating (usually EOS token)
        tool_template: The tool response template component
        user_template: The user template component
        user_template_with_tools: The user template with tool usage component
        assistant_template: The assistant template component
        global_policy: The global policy, controls the behavior of the template
        system_policy: The system message policy, controls the behavior of forming the system message
        tool_policy: The tool policy for the template, controls the behavior of forming tools.
    """
    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The template of the system prompt with tool usage
    system_template_with_tools: str = None
    # The system message
    system_message: str = ""
    # Behaviors
    # The tool template
    tool_template: str = None
    # The single tool observation template
    tool_observation_template: str = "{observation}"
    # The user template
    user_template: str = None
    user_template_with_tools: str = None
    # The assistant template
    assistant_template: str = None
    # The parallel tool calls template
    tool_calls_template: str = "{tool_calls}"
    # The single tool call template
    tool_call_template: str = "{tool_call}"


    # Stop criteria (the default one is EOS token)
    stop_words: Union[str, List[str]] = None
    # Generation prompt
    generation_prompt: str = None
    # Global policy
    global_policy: "GlobalPolicy" = None
    # System message policy
    system_policy: "SystemPolicy" = None
    # Assistant policy
    assistant_policy: "AssistantPolicy" = None
    # Tool policy for this template
    tool_policy: "ToolPolicy" = None

    ## vision part
    vision_start: str = None
    vision_end: str = None
    image_token: str = None
    video_token: str = None

    chat_template: str = None

    def __post_init__(self):
        """Post-initialization to automatically register vision processor if vision tokens are defined"""
        if self.image_token or self.video_token:
            self._register_vision_processor()
        # Initialise default tool policy if none was provided
        if self.tool_policy is None:
            self.tool_policy = ToolPolicy()
        if self.system_policy is None:
            self.system_policy = SystemPolicy()
        if self.assistant_policy is None:
            self.assistant_policy = AssistantPolicy()
    
    def _register_vision_processor(self):
        """Automatically register a vision processor for this template"""
        from ..vision import VisionProcessorConfig, register_processor
        
        # Determine model type based on template name
        model_type = self._infer_model_type()
        
        # Create vision config
        config = VisionProcessorConfig(
            model_type=model_type,
            image_token=self.image_token or "",
            video_token=self.video_token or "",
            vision_start=self.vision_start or "",
            vision_end=self.vision_end or "",
            processor_class="AutoProcessor",
            expansion_strategy="patch_based"
        )
        
        # Register the processor
        register_processor(self.name, config)
    
    def _infer_model_type(self) -> str:
        """Infer model type from template name"""
        name_lower = self.name.lower()
        
        if "qwen" in name_lower:
            return "qwen_vl"
        elif "llava" in name_lower:
            return "llava"
        elif "gemma" in name_lower:
            return "gemma3"
        elif "paligemma" in name_lower:
            return "paligemma"
        elif "internvl" in name_lower:
            return "internvl"
        elif "minicpm" in name_lower:
            return "minicpm"
        elif "mllama" in name_lower:
            return "mllama"
        elif "pixtral" in name_lower:
            return "pixtral"
        elif "video" in name_lower:
            return "video_llava"
        else:
            # Default to patch-based for unknown models
            return "patch_based"

    def _supports_tool_call(self) -> bool:
        if (self.system_template_with_tools or self.user_template_with_tools) and self.tool_template:
            return True
        else:
            return False

    def render(self, messages: List[Dict], tools=None, add_generation_prompt: bool = False) -> str:
        """
        Render the messages to a string prompt by applying the chat template. The render logic is implemented in the `Renderer` class.

        Args:
            messages: The list of messages
            tools: The list of tools
            add_generation_prompt: Whether to add the generation prompt

        Returns:
            The string formatted prompt for the messages after applying the chat template.
        """
        return Renderer(self).render(messages, tools, add_generation_prompt)

    def encode(self, messages: List[Dict], tokenizer: PreTrainedTokenizer, return_tensors: str = None, tools=None, add_generation_prompt=False, processor=None, **kwargs) -> str:
        """Encode the messages to token ids.

        Args:
            messages: The list of messages
            tokenizer: The tokenizer
            return_tensors: The return tensors
            tools: The list of tools
            add_generation_prompt: Whether to add the generation prefix
            processor: The processor for vision templates
        
        Returns:
            inputs: The dictionary of input ids, attention mask, labels, and action mask
        """
        if processor is None and self.supports_vision():
            raise ValueError(f"Processor is required for vision templates: {self.name}")
        
        if self.supports_vision():
            # Use vision-aware encoding with proper alignment
            return self._encode_with_vision_processor(messages, tokenizer, return_tensors, tools, add_generation_prompt=add_generation_prompt, processor=processor, **kwargs)
        else:
            # Use standard encoding
            return self._encode_standard(messages, tokenizer, return_tensors, tools, add_generation_prompt=add_generation_prompt, **kwargs)

    def _encode_standard(self, messages: List[Dict], tokenizer: PreTrainedTokenizer, return_tensors: str = None, tools=None, add_generation_prompt=False, **kwargs) -> str:
        logger.debug(f"[Template] Encoding standard for template: {self.name}")
        """Standard encoding without vision support"""
        prompt, elements, mask_flags = self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
        input_ids = []
        attention_mask = []
        labels = []
        action_mask = []

        if tokenizer.bos_token:
            # If add_bos_token is not set, we assume to add bos token
            # There is potential issue if the tokenizer has bos_token but do not add it by default
            if getattr(tokenizer, "add_bos_token", True):
                input_ids.append(tokenizer.bos_token_id)
                attention_mask.append(1)
                labels.append(-100)
                action_mask.append(0)
        
        for element, mask_flag in zip(elements, mask_flags):
            cur_input_ids = tokenizer.encode(element, add_special_tokens=False)
            input_ids.extend(cur_input_ids)
            attention_mask.extend([1] * len(cur_input_ids))
            if mask_flag:
                labels.extend([-100] * len(cur_input_ids))
                action_mask.extend([0] * len(cur_input_ids))
            else:
                labels.extend(cur_input_ids)
                action_mask.extend([1] * len(cur_input_ids))
        inputs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            action_mask=action_mask
        )
        if return_tensors == "pt":
            inputs = {k: torch.tensor([v]) for k, v in inputs.items()}
        return inputs

    def _encode_with_vision_processor(self, messages: List[Dict], tokenizer: PreTrainedTokenizer, return_tensors: str = None, tools=None, add_generation_prompt=False, processor=None, **kwargs) -> str:
        logger.debug(f"[Template] Encoding with vision processor for template: {self.name}")
        """Encode with vision processor handling proper alignment"""
        from ..vision import get_processor
        from ..utils import extract_vision_inputs_from_messages
        
        # Get vision processor
        vision_processor = get_processor(self.name)
        if vision_processor is None:
            raise ValueError(f"No vision processor registered for template: {self.name}")
        
        # Get base prompt and mask information
        prompt, elements, mask_flags = self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
        
        # Extract vision inputs
        images, videos = extract_vision_inputs_from_messages(messages)

        logger.debug(f"[Template] images: {len(images)}")
        logger.debug(f"[Template] videos: {len(videos)}")
        logger.debug(f"[Template] messages: {messages}")
        
        # Use vision processor with alignment support
        return vision_processor.process_for_llm(
            prompt=prompt,
            elements=elements,
            mask_flags=mask_flags,
            images=images,
            videos=videos,
            processor=processor,
            tokenizer=tokenizer,
            return_tensors=return_tensors
        )
    
    def supports_vision(self) -> bool:
        """Check if this template supports vision processing"""
        from ..vision import is_vision_template
        return is_vision_template(self.name)

    def get_vision_inputs(self, messages: List[Dict]):
        from ..utils.vision import open_image_from_any

        vision_inputs = defaultdict(list)
        logger.debug(f"[Template] get_vision_inputs: messages: {messages}")
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                for item in content:
                    if item['type'] == 'text':
                        continue
                    elif item['type'] in ['image', 'image_url', 'image_base64']:
                        vision_inputs["image"].append(open_image_from_any(item[item['type']]))
                    elif item['type'] == 'video':
                        raise NotImplementedError("Video is not supported for chat template.")
                    else:
                        raise ValueError(f"Invalid message type: {item['type']}")
            else:
                raise ValueError(f"Invalid message content: {content}, the content should be a list of dicts")
        return vision_inputs

    def jinja_template(self) -> str:
        """Get the Jinja template string for the template. The jinja template is compitable with Hugging Face's tokenizer.apply_chat_template method.
        
        Example:
        ```python
        tokenizer.chat_template = template.jinja_template()
        tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
        ```
        Returns:
            The Jinja template for the template.
        """
        if self.chat_template:
            return self.chat_template
        else:
            return JinjaGenerator(self).generate_jinja_template()

    def render_with_mask(self, messages: List[Dict], add_generation_prompt: bool = False, tools=None, **kwargs):
        from termcolor import colored
        prompt, elements, mask_flags = self.render(messages, add_generation_prompt=add_generation_prompt, tools=tools, **kwargs)

        prompt = ""
        for element, mask_flag in zip(elements, mask_flags):
            if mask_flag:
                prompt += colored(element, "red")
            else:
                prompt += colored(element, "green")
        return prompt, elements, mask_flags

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message


    def copy(self):
        return self.__class__(
            name=self.name,
            system_template=self.system_template,
            system_template_with_tools=self.system_template_with_tools,
            system_message=self.system_message,
            user_template=self.user_template,
            user_template_with_tools=self.user_template_with_tools,
            assistant_template=self.assistant_template,
            tool_calls_template=self.tool_calls_template,
            tool_call_template=self.tool_call_template,
            tool_template=self.tool_template,
            tool_observation_template=self.tool_observation_template,
            stop_words=self.stop_words,
            generation_prompt=self.generation_prompt,
            vision_start=self.vision_start,
            vision_end=self.vision_end,
            image_token=self.image_token,
            video_token=self.video_token,
            global_policy=deepcopy(self.global_policy),
            system_policy=deepcopy(self.system_policy),
            tool_policy=deepcopy(self.tool_policy),
            assistant_policy=deepcopy(self.assistant_policy),
            chat_template=self.chat_template,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "system_template_with_tools": self.system_template_with_tools,
            "stop_words": self.stop_words,
            "vision_start": self.vision_start,
            "vision_end": self.vision_end,
            "image_token": self.image_token,
            "video_token": self.video_token,
        }


class Qwen3Template(Template):
    def render(self, messages: List[Dict], tools=None, add_generation_prompt: bool = False, enable_thinking: bool = False) -> str:
        return Qwen3Renderer(self).render(messages, tools, add_generation_prompt, enable_thinking)
    

class HFTemplate(Template):
    """
    A general template that uses Hugging Face tokenizer's chat template.
    It aims to work for most HF's tokenizer's chat template and supports masking by applying chat template iteratively to 
    increasing turns of messages. This method only works for chat template that directly append elements when geting more
    turns of messages. It does not work for chat template that will modify the previous prompt of content (e.g. Qwen3's 
    template which deletes the previous thinking content).
    """
    def __init__(self, name: str):
        super().__init__(name=name)
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if self.tokenizer.chat_template is None:
            raise ValueError(f"Tokenizer from {name} does not have a chat_template. Cannot use HFTemplate.")
        # Use HFRenderer for rendering
        from .renderer import HFRenderer
        self._renderer = HFRenderer(self)

    def render(self, messages: List[Dict], tools=None, add_generation_prompt: bool = False, **kwargs) -> Tuple[str, List[str], List[bool]]:
        """Render messages using HF tokenizer's chat template.
        
        Returns:
            prompt: The final prompt string
            elements: The list of string *elements* that compose the prompt
            mask_flags: The list of mask flags for the elements
        """
        return self._renderer.render(messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
    
    def encode(self, messages: List[Dict], tokenizer: PreTrainedTokenizer, return_tensors: str = None, tools=None, add_generation_prompt=False, processor=None, **kwargs) -> Dict[str, Any]:
        """Encode messages by reusing render() to get elements and mask_flags, then tokenizing.
        
        This method follows the same pattern as Template._encode_standard() by:
        1. Calling render() to get (prompt, elements, mask_flags)
        2. Tokenizing each element separately
        3. Using mask_flags to set labels and action_mask
        
        Args:
            messages: The list of messages
            tokenizer: The tokenizer (should be the same as self.tokenizer, but kept for compatibility)
            return_tensors: The return tensors format ("pt" for PyTorch)
            tools: The list of tools
            add_generation_prompt: Whether to add the generation prefix
            processor: Not used for HF templates (kept for compatibility)
            **kwargs: Additional keyword arguments for render()
        
        Returns:
            inputs: Dictionary with input_ids, attention_mask, labels, and action_mask
        """
        # Use self.tokenizer if provided tokenizer is different (for compatibility)
        tokenizer_to_use = self.tokenizer
        
        # Reuse render() to get elements and mask_flags (same as base Template)
        prompt, elements, mask_flags = self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
        
        input_ids = []
        attention_mask = []
        labels = []
        action_mask = []
        
        # Handle BOS token if needed (same as base Template)
        if tokenizer_to_use.bos_token:
            if getattr(tokenizer_to_use, "add_bos_token", True):
                input_ids.append(tokenizer_to_use.bos_token_id)
                attention_mask.append(1)
                labels.append(-100)
                action_mask.append(0)
        
        # Tokenize each element separately (same as base Template)
        for element, mask_flag in zip(elements, mask_flags):
            cur_input_ids = tokenizer_to_use.encode(element, add_special_tokens=False)
            input_ids.extend(cur_input_ids)
            attention_mask.extend([1] * len(cur_input_ids))
            if mask_flag:
                # mask_flag=True means non-assistant, so mask it
                labels.extend([-100] * len(cur_input_ids))
                action_mask.extend([0] * len(cur_input_ids))
            else:
                # mask_flag=False means assistant, so don't mask it
                labels.extend(cur_input_ids)
                action_mask.extend([1] * len(cur_input_ids))
        
        inputs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            action_mask=action_mask
        )
        
        if return_tensors == "pt":
            inputs = {k: torch.tensor([v]) for k, v in inputs.items()}
        
        return inputs

    def jinja_template(self) -> str:
        return self.tokenizer.chat_template

if __name__ == "__main__":
    pass