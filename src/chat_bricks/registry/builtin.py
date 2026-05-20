from ..constants import ToolPlacement
from ..policies import (AssistantPolicy, GlobalPolicy, JsonCompactFormatter,
                        JsonFormatterNoBreakLine, JsonIndentedFormatter,
                        KimiK2ToolCallContentProcessor, Llama32DateProcessor,
                        Qwen25AssistantContentProcessor, SystemPolicy,
                        ToolMainContentProcessor, ToolPolicy)
from ..templates import Qwen3Template, Template
from . import register_template

register_template(
    Template(
        name="qwen2.5-no-system-tool",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
        observations_template="<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n",
        stop_words=["<|im_end|>"],
    )
)

register_template(
    Template(
        name="qwen2.5-vl",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        system_message="You are a helpful assistant.",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
        observations_template="<|im_start|>tool\n{observation}<|im_end|>\n",
        vision_start="<|vision_start|>",
        vision_end="<|vision_end|>",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        stop_words=["<|im_end|>"],
    )
)

register_template(
    Template(
        name="qwen2.5-vl-system-tool",
        system_template="<|im_start|>system\n{system_message}{tools}<|im_end|>\n",
        system_message="You are a helpful assistant.",
        tools_template="""\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>""",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
        observations_template="<|im_start|>tool\n{observation}<|im_end|>\n",
        vision_start="<|vision_start|>",
        vision_end="<|vision_end|>",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        stop_words=["<|im_end|>"],
    )
)

register_template(
    Template(
        name="qwen3-vl-instruct",
        system_template="<|im_start|>system\n{system_message}{tools}<|im_end|>\n",
        tools_template="""# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>""",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant{content}{tool_calls}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n",
        single_tool_call_template="\n<tool_call>\n{tool_call}\n</tool_call>",
        observations_template="<|im_start|>user{observations}<|im_end|>\n",
        single_observation_template="\n<tool_response>\n{observation}\n</tool_response>",
        vision_start="<|vision_start|>",
        vision_end="<|vision_end|>",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        stop_words=["<|im_end|>"],
        system_policy=SystemPolicy(
            use_system_without_system_message=False,
            content_processor=lambda system, tools: f"{system}\n\n"
            if (system != "" and tools)
            else system,
        ),
        assistant_policy=AssistantPolicy(
            content_processor=Qwen25AssistantContentProcessor(),
        ),
        chat_template="{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {%- if messages[0].content is string %}\n            {{- messages[0].content }}\n        {%- else %}\n            {%- for content in messages[0].content %}\n                {%- if 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' }}\n        {%- if messages[0].content is string %}\n            {{- messages[0].content }}\n        {%- else %}\n            {%- for content in messages[0].content %}\n                {%- if 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set image_count = namespace(value=0) %}\n{%- set video_count = namespace(value=0) %}\n{%- for message in messages %}\n    {%- if message.role == \"user\" %}\n        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content in message.content %}\n                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}\n                    <|vision_start|><|image_pad|><|vision_end|>\n                {%- elif content.type == 'video' or 'video' in content %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}\n                    <|vision_start|><|video_pad|><|vision_end|>\n                {%- elif 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content_item in message.content %}\n                {%- if 'text' in content_item %}\n                    {{- content_item.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and message.content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content in message.content %}\n                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}\n                    <|vision_start|><|image_pad|><|vision_end|>\n                {%- elif content.type == 'video' or 'video' in content %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}\n                    <|vision_start|><|video_pad|><|vision_end|>\n                {%- elif 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n",
    )
)

register_template(
    Template(
        name="qwen2.5",
        system_template="<|im_start|>system\n{system_message}{tools}<|im_end|>\n",
        system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        tools_template="""\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>""",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant{content}{tool_calls}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n",
        single_tool_call_template="\n<tool_call>\n{tool_call}\n</tool_call>",
        observations_template="<|im_start|>user{observations}<|im_end|>\n",
        single_observation_template="\n<tool_response>\n{observation}\n</tool_response>",
        stop_words=["<|im_end|>"],
        assistant_policy=AssistantPolicy(
            content_processor=Qwen25AssistantContentProcessor(),
        ),
    )
)


register_template(
    Template(
        name="qwen2.5-think",
        system_template="<|im_start|>system\n{system_message}{tools}<|im_end|>\n",
        system_message="You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.",
        # NOTE: the legacy ``system_template_with_tools`` for this template
        # entirely replaced ``system_message`` with a tool-aware preamble (and
        # used an unusual ``<|im_start|>`` without the ``system`` role label).
        # The new section-template pattern instead appends a tools block to the
        # normal system message, which is the cleaner behaviour. Tests that
        # asserted byte-for-byte on the legacy preamble will need updating.
        tools_template="""\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<think> [reasoning process here] </think>\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>\nYou must think first before calling any tool.""",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n<think>{content}<|im_end|>\n",
        observations_template="<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n",
        stop_words=["<|im_end|>"],
        vision_start="<|vision_start|>",
        vision_end="<|vision_end|>",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
    )
)

register_template(
    Qwen3Template(
        name="qwen3",
        system_template="<|im_start|>system\n{system_message}{tools}<|im_end|>\n",
        tools_template="""# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>""",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant{content}{tool_calls}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n",
        single_tool_call_template="\n<tool_call>\n{tool_call}\n</tool_call>",
        observations_template="<|im_start|>user{observations}<|im_end|>\n",
        single_observation_template="\n<tool_response>\n{observation}\n</tool_response>",
        stop_words=["<|im_end|>"],
        system_policy=SystemPolicy(
            use_system_without_system_message=False,
            content_processor=lambda system, tools: f"{system}\n\n"
            if (system != "" and tools)
            else system,
        ),
        assistant_policy=AssistantPolicy(
            content_processor=Qwen25AssistantContentProcessor(),
        ),
        chat_template="{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
    )
)

register_template(
    Template(
        name="qwen3-instruct",
        system_template="<|im_start|>system\n{system_message}{tools}<|im_end|>\n",
        tools_template="""# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>""",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant{content}{tool_calls}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n",
        single_tool_call_template="\n<tool_call>\n{tool_call}\n</tool_call>",
        observations_template="<|im_start|>user{observations}<|im_end|>\n",
        single_observation_template="\n<tool_response>\n{observation}\n</tool_response>",
        vision_start="<|vision_start|>",
        vision_end="<|vision_end|>",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        stop_words=["<|im_end|>"],
        system_policy=SystemPolicy(
            use_system_without_system_message=False,
            content_processor=lambda system, tools: f"{system}\n\n"
            if (system != "" and tools)
            else system,
        ),
        assistant_policy=AssistantPolicy(
            content_processor=Qwen25AssistantContentProcessor(),
        ),
        chat_template="{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {%- if messages[0].content is string %}\n            {{- messages[0].content }}\n        {%- else %}\n            {%- for content in messages[0].content %}\n                {%- if 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' }}\n        {%- if messages[0].content is string %}\n            {{- messages[0].content }}\n        {%- else %}\n            {%- for content in messages[0].content %}\n                {%- if 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set image_count = namespace(value=0) %}\n{%- set video_count = namespace(value=0) %}\n{%- for message in messages %}\n    {%- if message.role == \"user\" %}\n        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content in message.content %}\n                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}\n                    <|vision_start|><|image_pad|><|vision_end|>\n                {%- elif content.type == 'video' or 'video' in content %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}\n                    <|vision_start|><|video_pad|><|vision_end|>\n                {%- elif 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content_item in message.content %}\n                {%- if 'text' in content_item %}\n                    {{- content_item.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and message.content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content in message.content %}\n                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}\n                    <|vision_start|><|image_pad|><|vision_end|>\n                {%- elif content.type == 'video' or 'video' in content %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}\n                    <|vision_start|><|video_pad|><|vision_end|>\n                {%- elif 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n",
    )
)

register_template(
    Template(
        name="deepseek-prover",
        system_template="{system_message}\n",
        system_message="You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.",
        user_template="### Instruction:\n{content}\n",
        assistant_template="### Response:\n{content}\n<|EOT|>\n",
        stop_words=["<|EOT|>"],
    )
)


# TODO: mistral template has many cornor cases, leave it for now
# register_template(
#     Template(
#         name="mistral",
#         system_template="{system_message}",
#         user_template="[INST] {content}[/INST] ",
#         user_template_with_tools="[AVAILABLE TOOLS] {tools} [/AVAILABLE TOOLS] [INST] {content}[/INST] ",
#         assistant_template="{content}</s>",
#         observations_template="{observation}",
#         stop_words=["</s>"],
#         system_policy=SystemPolicy(
#             use_system=False,
#         ),
#         tool_policy=ToolPolicy(
#             placement=ToolPlacement.LAST_USER,
#             formatter=JsonCompactFormatter()
#         )
#     )
# )

# TODO: system template includes current date
register_template(
    Template(
        name="llama-3.2",
        # Tools placement is FIRST_USER, so the catalogue lives in the user
        # message via ``user_template_with_tools``. The ``{tools}`` slot in the
        # system_template only carries the ``Environment: ipython`` flag header
        # that legacy ``system_template_with_tools`` injected when tools exist.
        system_template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{tools}{system_message}<|eot_id|>",
        tools_template="Environment: ipython\n",
        user_template="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        user_template_with_tools="""<|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}.Do not use variables.\n\n{tools}\n\n{content}<|eot_id|>""",
        assistant_template="<|start_header_id|>assistant<|end_header_id|>\n\n{content}{tool_calls}<|eot_id|>",
        single_tool_call_template="{tool_call}",
        observations_template="""<|start_header_id|>ipython<|end_header_id|>\n\n"{observation}"<|eot_id|>""",
        stop_words=["<|eot_id|>"],
        system_policy=SystemPolicy(
            use_system=True,
            content_processor=Llama32DateProcessor(),
        ),
        tool_policy=ToolPolicy(
            placement=ToolPlacement.FIRST_USER, formatter=JsonIndentedFormatter()
        ),
    )
)

register_template(
    Template(
        name="glm-4",
        system_template="<|system|>\n{system_message}",
        user_template="<|user|>\n{content}",
        assistant_template="<|assistant|>\n{content}",
        stop_words=[""],
        global_policy=GlobalPolicy(prefix="[gMASK]<sop>"),
        system_policy=SystemPolicy(
            use_system=True,
            use_system_without_system_message=False,
        ),
    )
)

# GLM-4.5 / GLM-4.6 — full tool-call support, matches the chat_template
# shipped with `zai-org/GLM-4.5-Air` and `zai-org/GLM-4.5`. Differences vs
# `glm-4`:
#   - Tool catalogue gets its own block in the system message (`{tools}` slot)
#   - Assistant content is preceded by `<think></think>` (empty for non-thinking)
#   - Each tool call is wrapped in `<tool_call>{body}</tool_call>` markers
#   - Tool responses live in the `<|observation|>` role (NOT `<|user|>`),
#     wrapped in `<tool_response>...</tool_response>`
#   - `stop_words` lists every role marker so generation terminates correctly
register_template(
    Template(
        name="glm-4.5",
        system_template=(
            "<|system|>\n{system_message}{tools}"
        ),
        tools_template=(
            "\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n{tools}\n</tools>\n\n"
            "For each function call, return a json object with function name and "
            "arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
            "</tool_call>"
        ),
        user_template="<|user|>\n{content}",
        assistant_template="<|assistant|>\n<think></think>{content}{tool_calls}",
        generation_prompt="<|assistant|>\n<think></think>\n",
        single_tool_call_template="\n<tool_call>\n{tool_call}\n</tool_call>",
        observations_template="<|observation|>{observations}",
        single_observation_template="\n<tool_response>\n{observation}\n</tool_response>",
        stop_words=["<|user|>", "<|observation|>", "<|endoftext|>"],
        global_policy=GlobalPolicy(prefix="[gMASK]<sop>"),
        system_policy=SystemPolicy(
            use_system=True,
            use_system_without_system_message=True,
        ),
    )
)

register_template(
    Template(
        name="phi-4",
        system_template="<|im_start|>system<|im_sep|>{system_message}<|im_end|>",
        user_template="<|im_start|>user<|im_sep|>{content}<|im_end|>",
        assistant_template="<|im_start|>assistant<|im_sep|>{content}<|im_end|>",
        stop_words=["<|im_end|>"],
    )
)

# Note: Partial align, some minor new-line problems.
register_template(
    Template(
        name="nemotron",
        system_template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_message}{tools}<|eot_id|>",
        tools_template="<AVAILABLE_TOOLS>{tools}</AVAILABLE_TOOLS>",
        user_template="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        assistant_template="<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        observations_template="<|start_header_id|>user<|end_header_id|>\n\n<TOOL_RESPONSE>[{observation}]</TOOL_RESPONSE><|eot_id|>",
        stop_words=["<|eot_id|>"],
        system_policy=SystemPolicy(
            use_system=True,
            content_processor=lambda system_message, tools: f"\n{system_message}",
        ),
        tool_policy=ToolPolicy(
            placement=ToolPlacement.SYSTEM,
            content_processor=ToolMainContentProcessor(),
            formatter=JsonCompactFormatter(),
        ),
    )
)

register_template(
    Template(
        name="deepseek-r1-distill-qwen",
        system_template="{system_message}",
        user_template="<｜User｜>{content}",
        assistant_template="<｜Assistant｜>{content}<｜end▁of▁sentence｜>",
        stop_words=["<｜end▁of▁sentence｜>"],
        generation_prompt="<｜Assistant｜><think>\n",
        global_policy=GlobalPolicy(prefix="<｜begin▁of▁sentence｜>"),
        system_policy=SystemPolicy(
            use_system=True,
            use_system_without_system_message=False,
        ),
        chat_template="{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
    )
)

# DeepSeek-V3.1 — full tool-call support, matches the chat_template shipped
# with `deepseek-ai/DeepSeek-V3.1`. Differences vs `deepseek-r1-distill-qwen`:
#   - No `<think>` opener in generation_prompt — V3.1 uses `</think>` to
#     end an empty thinking block (non-thinking mode by default)
#   - Tool catalogue lives in the system prompt
#   - Each call is `{name}<｜tool▁sep｜>{args_body}` inside a
#     `<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>` wrapper; multiple calls
#     are grouped by `<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>`
#   - Tool responses live in their own `<｜tool▁output▁begin｜>...<｜tool▁output▁end｜>`
#     pair grouped by `<｜tool▁outputs▁begin｜>...<｜tool▁outputs▁end｜>`
#
# Body convention: `body_carries_name=False` — the function name lives
# in the marker via `<｜tool▁sep｜>`; the body following it is just args.
register_template(
    Template(
        name="deepseek-v3.1",
        system_template="{system_message}{tools}",
        tools_template=(
            "\n\n## Tools\nYou have access to the following tools:\n\n"
            "{tools}\n\n"
            "## Tool Use Rules\n"
            "When you need to call a tool, output it in the following format:\n"
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>NAME<｜tool▁sep｜>ARGUMENTS<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n"
            "where NAME is the function name and ARGUMENTS is a JSON object "
            "with the function arguments."
        ),
        user_template="<｜User｜>{content}",
        assistant_template="<｜Assistant｜></think>{content}{tool_calls}<｜end▁of▁sentence｜>",
        generation_prompt="<｜Assistant｜></think>",
        tool_calls_template="<｜tool▁calls▁begin｜>{tool_calls}<｜tool▁calls▁end｜>",
        single_tool_call_template="<｜tool▁call▁begin｜>{tool_call}<｜tool▁call▁end｜>",
        observations_template="<｜tool▁outputs▁begin｜>{observations}<｜tool▁outputs▁end｜>",
        single_observation_template="<｜tool▁output▁begin｜>{observation}<｜tool▁output▁end｜>",
        stop_words=["<｜end▁of▁sentence｜>"],
        global_policy=GlobalPolicy(prefix="<｜begin▁of▁sentence｜>"),
        system_policy=SystemPolicy(
            use_system=True,
            use_system_without_system_message=False,
        ),
    )
)

register_template(
    Template(
        name="llemma",
        system_template="{system_message}",
        user_template="Input:{content}\n\n",
        assistant_template="Response:{content}</s>",
        stop_words=["</s>"],
    )
)

register_template(
    Template(
        name="kimi-k2-instruct",
        # Kimi-K2 puts the tool catalogue BEFORE the system message block,
        # so the ``{tools}`` slot sits at the very start of ``system_template``.
        system_template="{tools}<|im_system|>system<|im_middle|>{system_message}<|im_end|>\n",
        tools_template="<|im_system|>tool_declare<|im_middle|>{tools}<|im_end|>",
        system_message="You are Kimi, an AI assistant created by Moonshot AI.",
        user_template="<|im_user|>user<|im_middle|>{content}<|im_end|>",
        assistant_template="<|im_assistant|>assistant<|im_middle|>{content}{tool_calls}<|im_end|>",
        tool_calls_template="<|tool_calls_section_begin|>{tool_calls}<|tool_calls_section_end|>",
        single_tool_call_template="<|tool_call_begin|>{tool_call}<|tool_call_end|>",
        observations_template="{observations}",
        single_observation_template="<|im_system|>tool<|im_middle|>## Return of \n{observation}<|im_end|>",
        vision_start="<|vision_start|>",
        vision_end="<|vision_end|>",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        stop_words=["<|im_end|>"],
        system_policy=SystemPolicy(
            use_system=True,
            use_system_without_system_message=True,
        ),
        tool_policy=ToolPolicy(
            formatter=JsonCompactFormatter(),
            tool_call_content_processor=KimiK2ToolCallContentProcessor(),
        ),
    )
)

# Skills-aware Qwen template demonstrating the new section-template pattern.
# Matches Qwen 3.5's stock chat-template structure verbatim for the tools
# section, plus an additional ``# Skills`` block in the same style.
# Reference: https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/chat_template.jinja
# Key facts that drove the design:
# - Section order: ``{tools}{skills}{system_message}`` — Qwen 3.5 appends the
#   user's system content AFTER the tools block (not before).
# - The format instruction + ``<IMPORTANT>`` reminder is REQUIRED. Without it,
#   Qwen 3.5 improvises a different tool-call format that vLLM's parser can't
#   recognize (e.g. ``<tool_code>foo(a=1)</tool_code>``).
# - The default ``ToolPolicy.formatter`` joins per-tool JSON with ``\n``, so the
#   ``{tools}`` placeholder lands as ``json1\njson2\n...`` inside ``<tools>``.
# - ``tools_template`` and ``skills_template`` end with ``\n\n`` so they slot
#   together cleanly when both are present, leaving exactly one blank line
#   before the user's system message.
register_template(
    Template(
        name="qwen-skills",
        system_template="<|im_start|>system\n{tools}{skills}{system_message}<|im_end|>\n",
        tools_template="""# Tools\n\nYou have access to the following functions:\n\n<tools>\n{tools}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\n""",
        skills_template="# Skills\n\nYou may also load one of the following skills via the load_skill tool. Each skill bundles instructions and (optionally) scripts/references that activate only after load_skill is called.\n\n<skills>\n{skills}\n</skills>\n\n",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n<think>{content}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n<think>",
        observations_template="<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n",
        stop_words=["<|im_end|>"],
    )
)


register_template(
    Template(
        name="toolgen-qwen2.5",
        system_template="<|im_start|>system\n{system_message}{tools}<|im_end|>\n",
        system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        tools_template="""\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>""",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant{content}{tool_calls}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n",
        single_tool_call_template="\n<tool_call>\n{tool_call}\n</tool_call>",
        observations_template="<|im_start|>user{observations}<|im_end|>\n",
        single_observation_template="\n<tool_response>\n{observation}\n</tool_response>",
        stop_words=["<|im_end|>"],
        tool_policy=ToolPolicy(
            formatter=JsonFormatterNoBreakLine(),
        ),
        assistant_policy=AssistantPolicy(
            content_processor=Qwen25AssistantContentProcessor(),
        ),
    )
)
