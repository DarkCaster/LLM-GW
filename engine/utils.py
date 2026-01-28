import logger


def parse_openai_request_content(request_data: dict) -> tuple[str, str, int, int]:
    log = logger.get_logger("")
    # for now we can only get content from `input` or `messages` arrays, depending on operation
    input = request_data.get("input")
    messages = request_data.get("messages")
    # other return values
    content_type = ""
    prompt = ""
    max_tokens = 1
    message_count = 0
    # Try parsing text from `input` field
    if isinstance(input, str):
        prompt = input
        content_type = "input_str"
        message_count = 1
    elif isinstance(input, list):
        content_type = "input_list"
        for item in input:
            if isinstance(item, str):
                prompt += item
                message_count += 1
    elif isinstance(messages, list):
        content_type = "messages"
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                prompt += content
                message_count += 1
            elif isinstance(content, list):
                # Handle multi-modal content arrays
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        prompt += item.get("text", "")
                        message_count += 1
        # Get max_tokens field from request_data
        max_tokens = request_data.get("max_tokens")
        if max_tokens is None:
            max_tokens = request_data.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = 4096
            log.warning(
                f"No max_tokens or max_completion_tokens in request, defaulting to {max_tokens}"
            )
    else:
        message_count = 1
        log.error("No supported data for tokenization found in request")
    return content_type, prompt, max_tokens, message_count
