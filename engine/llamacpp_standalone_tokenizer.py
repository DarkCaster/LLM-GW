import asyncio
import json
import os
from typing import List
from .standalone_tokenizer import StandaloneTokenizer


class LlamaStandaloneTokenizer(StandaloneTokenizer):
    def __init__(self, add_tokens_per_message: int, binary_path: str, args: List[str]):
        super().__init__()
        self._add_tokens_per_message = add_tokens_per_message
        self._binary_path = binary_path
        self._args = args
        self.logger.debug(
            f"Initialized LlamaStandaloneTokenizer with binary_path: {self._binary_path}"
        )

    async def estimate_tokens(self, request_data: dict) -> int:
        # Get max_tokens field from request_data
        max_tokens = request_data.get("max_tokens")
        if max_tokens is None:
            max_tokens = request_data.get("max_completion_tokens")
        if max_tokens is None:
            self.logger.warning(
                "No max_tokens or max_completion_tokens in request, defaulting to 4096"
            )
            max_tokens = 4096
        # Get messages from request_data
        messages = request_data.get("messages")
        if messages is None:
            self.logger.error("No messages field in request_data")
            return max_tokens + 32
        # Calculate messages count
        messages_count = len(messages)
        # Combine all message-contents together into single combined string
        combined_string = ""
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                combined_string += content + "\n"
            elif isinstance(content, list):
                # Handle multi-modal content arrays
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        combined_string += item.get("text", "") + "\n"

        # Debug: write combined_string to file for inspection
        # try:
        #     with open("LlamaStandaloneTokenizer.dump.txt", "w", encoding="utf-8") as f:
        #         f.write(combined_string)
        #     self.logger.debug(
        #         f"Wrote combined_string ({len(combined_string)} chars) to LlamaStandaloneTokenizer.dump.txt"
        #     )
        # except Exception as e:
        #    self.logger.warning(f"Failed to write debug dump file: {e}")

        # Get workdir from binary base path
        workdir = os.path.dirname(os.path.abspath(self._binary_path))
        # Run llama-tokenizer process with provided args, send combined string to process stdin
        try:
            self.logger.info(
                f"Running process: {self._binary_path} with args: {self._args}"
            )
            process = await asyncio.subprocess.create_subprocess_exec(
                self._binary_path,
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
            )
            # Send combined string to stdin and wait for process to complete
            stdout, stderr = await process.communicate(
                input=combined_string.encode("utf-8")
            )
            # Log stderr if present
            if stderr:
                stderr_str = stderr.decode("utf-8", errors="replace").strip()
                if stderr_str:
                    self.logger.warning(f"Tokenizer stderr: {stderr_str}")
            # Read result from stdout
            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            # Parse the JSON-like number array format containing tokens "[24048, 198, n, ...]"
            try:
                tokens_array = json.loads(stdout_str)
                if not isinstance(tokens_array, list):
                    self.logger.error(f"Tokenizer output is not a list: {stdout_str}")
                    return max_tokens + 32
                # Calculate token count in array
                token_count = len(tokens_array)
                self.logger.debug(
                    f"Tokenizer returned {token_count} tokens for {messages_count} messages"
                )
            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Failed to parse tokenizer output as JSON: {e}, output: {stdout_str}"
                )
                return max_tokens + 32
        except Exception as e:
            self.logger.error(f"Error running tokenizer process: {e}")
            return max_tokens + 32
        total_tokens = token_count + max_tokens + 32
        total_tokens += messages_count * self._add_tokens_per_message
        self.logger.debug(
            f"Token estimation: prompt={token_count}, max_tokens={max_tokens}, "
            f"messages_count={messages_count}, extra_per_message={self._add_tokens_per_message}, "
            f"total={total_tokens}"
        )
        return total_tokens
