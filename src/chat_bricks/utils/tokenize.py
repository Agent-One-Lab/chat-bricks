import logging

import torch

from ..registry import get_template

logger = logging.getLogger(__name__)


def transform_multi_turn_reward_mask(action_mask):
    """
    Given a binary action_mask of shape (batch_size, sequence_length),
    returns a tensor of the same shape with 1 only at the position where the action_mask is 1 and the next position is 0,
    """
    # action_mask: shape (batch_size, sequence_length)
    batch_size, seq_length = action_mask.shape

    # Create a shifted version of the attention mask by shifting left.
    # For the last column, we append a column of zeros.
    shifted = torch.cat(
        [
            action_mask[:, 1:],
            torch.zeros(
                batch_size, 1, dtype=action_mask.dtype, device=action_mask.device
            ),
        ],
        dim=1,
    )

    # Identify positions where the attention_mask is 1 and the shifted mask is 0.
    # This means either the next position is 0 or we're at the last element.
    last_ones_mask = (action_mask == 1) & (shifted == 0)

    # Optionally, convert boolean mask to integers (0s and 1s).
    return last_ones_mask.int()


def transform_reward_mask(action_mask):
    """
    Given a binary attention_mask of shape (batch_size, sequence_length),
    returns a tensor of the same shape with 1 only at the rightmost (last) 1 per row,
    and 0 everywhere else.
    """
    batch_size, seq_length = action_mask.shape

    # Check for rows that contain at least one 1.
    has_one = action_mask.sum(dim=1) > 0

    # Reverse each row so that the first occurrence of 1 corresponds to the last 1 in the original.
    reversed_mask = action_mask.flip(dims=[1])

    # For each row, find the index of the first occurrence of 1 in the reversed row.
    # Note: torch.argmax returns 0 if no element is 1, so we will handle rows with no ones separately.
    first_one_idx_reversed = torch.argmax(reversed_mask, dim=1)

    # Convert to the original index position.
    last_indices = seq_length - 1 - first_one_idx_reversed

    # Create an output tensor initialized with zeros.
    output = torch.zeros_like(action_mask)

    # For rows that have at least one 1, set the found last index to 1.
    # We use advanced indexing to assign 1 to the appropriate positions.
    row_indices = torch.arange(batch_size)
    output[row_indices[has_one], last_indices[has_one]] = 1

    return output


def tokenize_conversation(
    messages,
    tokenizer,
    template,
    max_length,
    tools=None,
    processor=None,
    return_tensors="pt",
    ignore_tool_calls=False,
    train_on_last_turn_only=False,
    **kwargs,  # Additional kwargs for the chat template, e.g. enable_thinking
):
    """
    We want to tokenize the whole conversation. But we can't just simply
    use get_prompt to get string prompt and tokenize it. Because the loss
    can only be computed on model's response. We want:
        input_ids
        attention_mask
        labels: should be -100 for user prompt and input id for model's response
        action_mask: should be 0 for user prompt and 1 for model's response

    Args:
        messages: The list of messages
        tokenizer: The tokenizer
        template: The template
        max_length: The maximum length of the input
        tools: The tools
        processor: The processor
        return_tensors: The return tensors
        **kwargs: Additional kwargs for the chat template, e.g. enable_thinking

    Returns:
        inputs: The dictionary of input ids, attention mask, labels, and action mask
    """
    from .. import Chat

    chat = Chat(
        template=template,
        messages=messages,
        tokenizer=tokenizer,
        ignore_tool_calls=ignore_tool_calls,
    )
    inputs = chat.tokenize(tokenizer, tools=tools, processor=processor, train_on_last_turn_only=train_on_last_turn_only, **kwargs)

    if max_length is not None:
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
        if "labels" in inputs:
            inputs["labels"] = inputs["labels"][:, :max_length]
        if "action_mask" in inputs:
            inputs["action_mask"] = inputs["action_mask"][:, :max_length]

    return inputs


def tokenize_conversations(
    messages_list,
    tokenizer,
    template,
    max_length,
    processor=None,
    return_tensors="pt",
    return_reward_mask=False,
    padding_side="right",
    concatenate_mm_inputs=False,
    ignore_tool_calls=False,
    train_on_last_turn_only=False,
    **kwargs,
):
    batch_input_ids = []
    batch_attention_masks = []
    batch_labels = []
    batch_action_masks = []
    batch_mm_inputs = []
    # TODO: add multiprocessing
    template = get_template(template)

    for i, messages in enumerate(messages_list):
        # logger.info(f"[chat-bricks/tokenize_conversations] Tokenizing conversation {i+1} of {len(messages_list)}")
        # logger.info(f"[chat-bricks/tokenize_conversations] Messages: {messages}")
        inputs = tokenize_conversation(
            messages=messages,
            tokenizer=tokenizer,
            template=template,
            max_length=max_length,
            processor=processor,
            return_tensors=return_tensors,
            ignore_tool_calls=ignore_tool_calls,
            train_on_last_turn_only=train_on_last_turn_only,
            **kwargs,
        )
        batch_input_ids.append(inputs["input_ids"].squeeze(0))
        batch_attention_masks.append(inputs["attention_mask"].squeeze(0))
        batch_labels.append(inputs["labels"].squeeze(0))
        batch_action_masks.append(inputs["action_mask"].squeeze(0))
        mm_inputs = {}
        if "pixel_values" in inputs:
            mm_inputs["pixel_values"] = inputs["pixel_values"]
        else:
            mm_inputs["pixel_values"] = None
        if "image_grid_thw" in inputs:
            mm_inputs["image_grid_thw"] = inputs["image_grid_thw"]
        else:
            mm_inputs["image_grid_thw"] = None

        batch_mm_inputs.append(mm_inputs)

    if return_tensors == "pt":
        # Use pad_token_id from the tokenizer interface
        pad_token_id = getattr(tokenizer, "pad_token_id", 0)

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=pad_token_id,
            padding_side=padding_side,
        )
        batch_attention_masks = torch.nn.utils.rnn.pad_sequence(
            batch_attention_masks,
            batch_first=True,
            padding_value=0,
            padding_side=padding_side,
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels,
            batch_first=True,
            padding_value=-100,
            padding_side=padding_side,
        )
        batch_action_masks = torch.nn.utils.rnn.pad_sequence(
            batch_action_masks,
            batch_first=True,
            padding_value=0,
            padding_side=padding_side,
        )

    # convert [{"pixel_values": tensor, "image_grid_thw": tensor}, ...] to {"key1":  concat_tensor, "key2": concat_tensor, ...}
    concatenated_mm_inputs = {}
    if concatenate_mm_inputs:
        for key in batch_mm_inputs[0].keys():
            if isinstance(mm_inputs[key], torch.Tensor):
                concatenated_mm_inputs[key] = torch.cat(
                    [
                        mm_inputs[key]
                        for mm_inputs in batch_mm_inputs
                        if mm_inputs[key] is not None
                    ],
                    dim=0,
                )

    inputs = dict(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_masks,
        labels=batch_labels,
        action_mask=batch_action_masks,
    )

    if return_reward_mask:
        inputs["reward_mask"] = transform_reward_mask(batch_action_masks)

    # Check if we need mm_inputs
    mm_keys = list(batch_mm_inputs[0].keys())
    return_mm_inputs = False
    for key in mm_keys:
        if any(mm_inputs[key] is not None for mm_inputs in batch_mm_inputs):
            return_mm_inputs = True
            break

    if return_mm_inputs:
        if concatenate_mm_inputs:
            inputs.update(concatenated_mm_inputs)
        else:
            inputs["mm_inputs"] = batch_mm_inputs

    return inputs
