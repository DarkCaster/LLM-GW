# engine/llamacpp_engine.py

from typing import List
import aiohttp
from .engine_client import EngineClient


class LlamaCppEngine(EngineClient):
    """
    Concrete implementation of EngineClient for llama.cpp engines.
    """

    def __init__(self, base_url: str):
        """
        Initialize the llama.cpp engine client.

        Args:
            base_url: Base URL for the llama.cpp server's HTTP endpoint
        """
        super().__init__(base_url)

    async def estimate_tokens(self, request_data: dict) -> int:
        """
        Calculate token requirements from incoming request using llama.cpp tokenization endpoint.

        Args:
            request_data: The incoming request data dictionary

        Returns:
            Estimated number of tokens required for this request (prompt + max_tokens)

        Raises:
            Exception: If token estimation fails
        """
        try:
            # Determine request type and extract content for tokenization
            is_chat = "messages" in request_data

            if is_chat:
                # Chat completions - extract messages
                messages = request_data.get("messages", [])
                # Build a simple text representation of the chat
                # llama.cpp tokenize endpoint typically accepts a prompt string
                text_parts = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    text_parts.append(f"{role}: {content}")
                prompt_text = "\n".join(text_parts)
            else:
                # Text completions - extract prompt
                prompt = request_data.get("prompt", "")
                if isinstance(prompt, list):
                    prompt_text = " ".join(prompt)
                else:
                    prompt_text = str(prompt)

            # Make request to llama.cpp tokenization endpoint
            tokenize_url = f"{self.base_url}/tokenize"

            payload = {"content": prompt_text}

            timeout = aiohttp.ClientTimeout(total=30.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(tokenize_url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Tokenization request failed with status {response.status}: {error_text}"
                        )

                    result = await response.json()

                    # llama.cpp returns tokens in a 'tokens' array
                    tokens = result.get("tokens", [])
                    prompt_tokens = len(tokens)

            # Add max_tokens if present in the request
            max_tokens = request_data.get("max_tokens", 0)

            total_tokens = prompt_tokens + max_tokens

            self.logger.debug(
                f"Estimated tokens: {prompt_tokens} (prompt) + {max_tokens} (max) = {total_tokens}"
            )

            return total_tokens

        except aiohttp.ClientError as e:
            self.logger.error(f"Failed to estimate tokens (network error): {e}")
            raise Exception(f"Token estimation failed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to estimate tokens: {e}")
            raise

    def transform_request(self, request_data: dict) -> dict:
        """
        Transform request data for llama.cpp-specific format.
        Remove or transform unsupported fields for llama.cpp.

        Args:
            request_data: Original request data dictionary

        Returns:
            Transformed request data dictionary
        """
        # Create a copy to avoid modifying the original
        transformed = request_data.copy()

        # List of OpenAI-specific fields that llama.cpp might not support
        # Note: llama.cpp actually supports many OpenAI fields, but we'll handle known incompatibilities
        unsupported_fields = []

        # Check for fields that might need special handling
        # For now, we'll be permissive and let llama.cpp handle most fields
        # Log warnings for fields that are commonly unsupported

        potentially_unsupported = [
            "user",
            "logit_bias",
            "functions",
            "function_call",
            "tools",
            "tool_choice",
        ]

        for field in potentially_unsupported:
            if field in transformed:
                unsupported_fields.append(field)
                # Remove the field
                del transformed[field]

        if unsupported_fields:
            self.logger.warning(
                f"Removed unsupported fields for llama.cpp: {unsupported_fields}"
            )

        return transformed

    def transform_response(self, response_data: dict) -> dict:
        """
        Transform llama.cpp response to OpenAI-compatible format.

        Args:
            response_data: Engine response data dictionary

        Returns:
            Transformed response data dictionary
        """
        # llama.cpp typically returns OpenAI-compatible responses already
        # We may need to add or adjust some fields for full compatibility

        # For now, pass through as llama.cpp is already OpenAI-compatible
        # Future enhancements can add field mapping if needed

        return response_data

    def get_supported_endpoints(self) -> List[str]:
        """
        Get list of supported API endpoints for llama.cpp.

        Returns:
            List of supported endpoint paths
        """
        return ["/v1/chat/completions", "/v1/completions"]

    async def check_health(self, timeout: float = 5.0) -> bool:
        """
        Check if llama.cpp engine endpoint is available and responding.
        Override to use llama.cpp-specific health check if available.

        Args:
            timeout: Timeout in seconds for the health check

        Returns:
            True if engine is healthy and responding, False otherwise
        """
        # llama.cpp has a /health endpoint, try that first, then fall back to base implementation
        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                # Try /health endpoint
                try:
                    async with session.get(f"{self.base_url}/health") as response:
                        if response.status == 200:
                            self.logger.debug(
                                f"Llama.cpp health check passed for {self.base_url}"
                            )
                            return True
                except aiohttp.ClientError:
                    pass

                # Fallback to /v1/models endpoint
                try:
                    async with session.get(f"{self.base_url}/v1/models") as response:
                        if response.status == 200:
                            self.logger.debug(
                                f"Llama.cpp health check passed (via /v1/models) for {self.base_url}"
                            )
                            return True
                except aiohttp.ClientError:
                    pass

                self.logger.debug(f"Llama.cpp health check failed for {self.base_url}")
                return False

        except Exception as e:
            self.logger.debug(f"Llama.cpp health check error for {self.base_url}: {e}")
            return False
