from .chat import Chat
from .constants import Role, ToolPlacement
from .policies import (AssistantPolicy, GlobalPolicy, JsonCompactFormatter,
                       JsonFormatter, JsonFormatterNoBreakLine,
                       JsonIndentedFormatter, JsonMinifiedFormatter,
                       JsonQwenFormatter, Llama32DateProcessor,
                       Qwen25AssistantContentProcessor, SystemContentProcessor,
                       SystemPolicy, ToolContentProcessor, ToolFormatter,
                       ToolMainContentProcessor, ToolPolicy)
from .registry import get_template, register_template
from .templates import HFTemplate, Qwen3Template, Template
from .utils import (compare_hf_template, display_messages, image_to_data_uri,
                    tokenize_conversation, tokenize_conversations,
                    validate_messages_for_template, split_messages_with_assistant)

# Vision symbols are loaded lazily so that importing chat_bricks does not pull
# in torch (vision_processor depends on it). Install the [train] extra to get
# torch; vision use will then work transparently via this __getattr__ hook.
_LAZY_VISION = {"VisionProcessor", "VisionProcessorConfig", "register_processor"}


def __getattr__(name):
    if name in _LAZY_VISION:
        from . import vision as _vision
        return getattr(_vision, name)
    raise AttributeError(f"module 'chat_bricks' has no attribute {name!r}")

__all__ = [
    "Chat",
    "AssistantPolicy",
    "GlobalPolicy",
    "JsonFormatter",
    "Llama32DateProcessor",
    "Qwen25AssistantContentProcessor",
    "SystemPolicy",
    "SystemContentProcessor",
    "ToolPolicy",
    "ToolMainContentProcessor",
    "ToolContentProcessor",
    "get_template",
    "register_template",
    "Template",
    "split_messages_with_assistant",
    "HFTemplate",
    "Qwen3Template",
    "VisionProcessor",
    "VisionProcessorConfig",
    "register_processor",
    "compare_hf_template",
    "tokenize_conversation",
    "tokenize_conversations",
    "validate_messages_for_template",
    "display_messages",
    "image_to_data_uri",
    "ToolPlacement",
    "Role",
    "JsonCompactFormatter",
    "JsonFormatterNoBreakLine",
    "JsonIndentedFormatter",
    "JsonMinifiedFormatter",
    "JsonQwenFormatter",
    "ToolFormatter",
]
