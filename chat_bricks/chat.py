from typing import List, Dict, Union, Any
from .registry import get_template
from transformers import PreTrainedTokenizer

class Chat:
    def __init__(self, template: str, messages: List[List[str]]=None, tools=None, tokenizer: PreTrainedTokenizer = None):
        """
        Args:
            template: The name of the template to use.
            messages: The messages to use for the chat.
            tools: The tools to use for the chat.
            tokenizer: The tokenizer to use for the chat.
        """
        self.template = get_template(template)
        self.messages = self.convert_to_hf_format_messages(messages)
        self.tokenizer = tokenizer
        self.tools = tools
        self.flags = {}

    def _detect_labels(self, messages):
        message = messages[0]
        if 'role' in message and "content" in message:
            return 'role', 'content'
        elif 'from' in message and "value" in message:
            return 'from', 'value'
        else:
            raise ValueError(f"Cannot find role label and content label in the data.")

    
    def _convert_single_message_to_hf_format(self, message: Dict) -> Dict:
        if isinstance(message['content'], str):
            message['content'] = [{"type": "text", "text": message['content']}]
        elif isinstance(message['content'], list):
            for item in message['content']:
                if item['type'] == 'text':
                    continue
                elif item['type'] in ["image", "image_url"]:
                    pass
                else:
                    raise ValueError(f"Invalid message type: {item['type']}")
                

    def convert_to_hf_format_messages(self, messages: Union[List[Dict], Dict[str, List[Dict]]]) -> List[Dict]:
        hf_messages = []
        if messages is None:
            return None
        role_label, content_label = self._detect_labels(messages)
        for message in messages:
            hf_message = {"role": message[role_label], "content": message[content_label]}
            if "tool_calls" in message:
                hf_message["tool_calls"] = message["tool_calls"]
            hf_messages.append(hf_message)
        
        for message in hf_messages:
            self._convert_single_message_to_hf_format(message)

        return hf_messages

    def set_messages(self, messages: List[Dict]):
        """Set the messages for the chat."""
        self.messages = self.convert_to_hf_format_messages(messages)

    def prompt(self, add_generation_prompt=False, tools=None, **kwargs) -> str:
        """Get the prompt for the chat.

        Args:
            add_generation_prompt: Whether to add the generation prompt.
            tools: The tools to use for the chat.
            **kwargs: Additional keyword arguments to pass to the template render method.

        Returns:
            The prompt for the chat.
        """
        self.flags['add_generation_prompt'] = add_generation_prompt
        tools = tools or self.tools
        prompt, _, _ = self.template.render(messages=self.messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
        return prompt

    def prompt_with_mask(self, add_generation_prompt=False, tools=None, **kwargs) -> str:
        tools = tools or self.tools
        prompt_with_mask, _, _ = self.template.render_with_mask(messages=self.messages, add_generation_prompt=add_generation_prompt, tools=tools, **kwargs)
        return prompt_with_mask

    def vision_inputs(self) -> List[Any]:
        return self.template.get_vision_inputs(self.messages)

    def tokenize(self, tokenizer: PreTrainedTokenizer = None, add_generation_prompt=False, tools=None, processor=None, **kwargs) -> List[int]:
        """Tokenize the messages.

        Args:
            tokenizer: The tokenizer to use for the chat.
            add_generation_prompt: Whether to add the generation prompt.
            tools: The tools to use for the chat.
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
                raise ValueError("Tokenizer is not set. Set it when initializing the chat or pass it as an argument.")
            tokenizer = self.tokenizer

        if tools is None:
            tools = self.tools
        return self.template.encode(messages=self.messages, tokenizer=tokenizer, return_tensors="pt", tools=tools, add_generation_prompt=add_generation_prompt, processor=processor, **kwargs)

    def append(self, message: Union[Dict]):
        self._convert_single_message_to_hf_format(message)
        self.messages.append(message)
