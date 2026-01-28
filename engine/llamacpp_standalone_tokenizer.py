import asyncio
import json
import os
from typing import List
from .standalone_tokenizer import StandaloneTokenizer
from .utils import parse_openai_request_content


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
        # Parse request
        try:
            _, prompt, max_tokens, message_count = parse_openai_request_content(
                request_data
            )
        except Exception as e:
            self.logger.error(f"Error parsing request_data: {e}")
            return 1
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
            # Send content prompt to stdin and wait for process to complete
            stdout, stderr = await process.communicate(
                input=prompt.encode("utf-8")
            )
            # Log stderr if present
            if stderr:
                stderr_str = stderr.decode("utf-8", errors="replace").strip()
                if stderr_str:
                    self.logger.warning(f"Tokenizer stderr: {stderr_str}")
            # Read result from stdout
            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            # Parse the JSON-like number array format containing tokens "[24048, 198, n, ...]"
            # trim all text before last `[` character in stdout_str, save result to stdout_str
            start_idx = stdout_str.rfind("[")
            if start_idx == -1:
                self.logger.error("No '[' found in tokenizer output")
                return max_tokens
            stdout_str = stdout_str[start_idx:]
            # trim all text after first `]` character in stdout_str, save result to stdout_str
            end_idx = stdout_str.find("]")
            if end_idx == -1:
                self.logger.error("No ']' found in tokenizer output")
                return max_tokens
            stdout_str = stdout_str[: end_idx + 1]
            try:
                tokens_array = json.loads(stdout_str)
                if not isinstance(tokens_array, list):
                    self.logger.error(f"Tokenizer output is not a list: {stdout_str}")
                    return max_tokens
                # Calculate token count in array
                token_count = len(tokens_array)
                self.logger.debug(
                    f"Tokenizer returned {token_count} tokens for {message_count} messages"
                )
            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Failed to parse tokenizer output as JSON: {e}, output: {stdout_str}"
                )
                return max_tokens
        except Exception as e:
            self.logger.error(f"Error running tokenizer process: {e}")
            return max_tokens
        total_tokens = token_count + max_tokens
        total_tokens += message_count * self._add_tokens_per_message
        self.logger.debug(
            f"Token estimation: prompt={token_count}, max_tokens={max_tokens}, "
            f"message_count={message_count}, extra_per_message={self._add_tokens_per_message}, "
            f"total={total_tokens}"
        )
        return total_tokens
