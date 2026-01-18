"""
LlamaCppEngine - llama.cpp specific implementation of EngineClient.

Provides llama.cpp specific token estimation, request transformation,
and response transformation functionality.
"""

import asyncio
import json
import logging
from typing import List, Optional

import aiohttp

from .engine_client import EngineClient


class LlamaCppEngine(EngineClient):
    """Concrete implementation of EngineClient for llama.cpp engines."""

    def __init__(self, base_url: str, logger: Optional[logging.Logger] = None):
        """
        Initialize LlamaCppEngine with llama.cpp specific configuration.

        Args:
            base_url: Base URL for llama.cpp HTTP endpoint
            logger: Logger instance (uses class name if not provided)
        """
        super().__init__(base_url, logger)

    async def estimate_tokens(self, request_data: dict) -> int:
        """
        Calculate token requirements from incoming request for llama.cpp.

        Args:
            request_data: Request data dictionary

        Returns:
            Estimated token count

        Raises:
            RuntimeError: If tokenization fails or engine is not available
            ValueError: If request format is invalid
        """
        try:
            # Determine request type and build appropriate payload
            endpoint = request_data.get("endpoint", "/v1/chat/completions")

            if endpoint == "/v1/chat/completions":
                prompt_tokens = await self._estimate_chat_tokens(request_data)
            elif endpoint == "/v1/completions":
                prompt_tokens = await self._estimate_text_tokens(request_data)
            else:
                raise ValueError(
                    f"Unsupported endpoint for token estimation: {endpoint}"
                )

            # Add max_tokens if present in request
            max_tokens = request_data.get("max_tokens", 0)
            total_tokens = prompt_tokens + max_tokens

            self.logger.debug(
                f"Token estimation: {prompt_tokens} prompt + {max_tokens} max = {total_tokens} total"
            )
            return total_tokens

        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error during token estimation: {e}")
            raise RuntimeError(f"Failed to estimate tokens: {e}")
        except (KeyError, ValueError) as e:
            self.logger.error(f"Invalid request format for token estimation: {e}")
            raise ValueError(f"Invalid request format: {e}")

    async def _estimate_chat_tokens(self, request_data: dict) -> int:
        """
        Estimate tokens for chat completions request.

        Args:
            request_data: Chat completions request data

        Returns:
            Estimated prompt token count
        """
        messages = request_data.get("messages", [])

        # Build prompt string from messages
        prompt_parts = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role and content:
                prompt_parts.append(f"{role}: {content}")

        prompt = "\n".join(prompt_parts)
        return await self._tokenize_text(prompt)

    async def _estimate_text_tokens(self, request_data: dict) -> int:
        """
        Estimate tokens for text completions request.

        Args:
            request_data: Text completions request data

        Returns:
            Estimated prompt token count
        """
        prompt = request_data.get("prompt", "")
        if isinstance(prompt, list):
            # Join multiple prompts
            prompt = " ".join(prompt)

        return await self._tokenize_text(str(prompt))

    async def _tokenize_text(self, text: str) -> int:
        """
        Tokenize text using llama.cpp tokenization endpoint.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens in the text

        Raises:
            RuntimeError: If tokenization fails
        """
        if not text:
            return 0

        url = f"{self.base_url}/tokenize"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json={"content": text},
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"Tokenization failed with status {response.status}: {error_text}"
                        )

                    result = await response.json()
                    tokens = result.get("tokens", [])
                    return len(tokens)

            except asyncio.TimeoutError:
                self.logger.error(
                    f"Tokenization timeout for text of length {len(text)}"
                )
                raise RuntimeError("Tokenization request timed out")
            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Invalid JSON response from tokenization endpoint: {e}"
                )
                raise RuntimeError("Invalid tokenization response format")

    def transform_request(self, request_data: dict) -> dict:
        """
        Transform OpenAI request to llama.cpp compatible format.

        Args:
            request_data: Original OpenAI request data

        Returns:
            Transformed request data for llama.cpp
        """
        transformed = request_data.copy()

        # Remove OpenAI-specific fields that llama.cpp doesn't support
        unsupported_fields = ["logit_bias", "logprobs", "top_logprobs", "user"]

        for field in unsupported_fields:
            if field in transformed:
                self.logger.warning(f"Removing unsupported field: {field}")
                del transformed[field]

        # Transform any field names if needed
        # Currently, llama.cpp uses the same field names as OpenAI API

        self.logger.debug(f"Transformed request: removed {unsupported_fields}")
        return transformed

    def transform_response(self, response_data: dict) -> dict:
        """
        Transform llama.cpp response to OpenAI compatible format.

        Args:
            response_data: Original llama.cpp response data

        Returns:
            Transformed response data in OpenAI format
        """
        transformed = response_data.copy()

        # Check if this is a streaming response
        is_streaming = transformed.get("stream", False)

        if is_streaming:
            # For streaming responses, we need to transform each chunk
            # The basic structure is already OpenAI compatible
            # We'll just ensure required fields are present
            if "choices" in transformed and transformed["choices"]:
                choice = transformed["choices"][0]
                if "text" in choice and "content" not in choice:
                    # Convert text to content for chat completions
                    choice["content"] = choice.pop("text")
        else:
            # For non-streaming responses, ensure the format matches OpenAI
            if "choices" in transformed and transformed["choices"]:
                for choice in transformed["choices"]:
                    # Ensure message structure for chat completions
                    if "message" not in choice and "text" in choice:
                        choice["message"] = {
                            "role": "assistant",
                            "content": choice.pop("text"),
                        }
                    elif "message" in choice and "content" not in choice["message"]:
                        # Ensure content field exists
                        choice["message"]["content"] = ""

        # Ensure required top-level fields
        if "object" not in transformed:
            transformed["object"] = "chat.completion"

        self.logger.debug("Transformed response to OpenAI format")
        return transformed

    def get_supported_endpoints(self) -> List[str]:
        """
        Get list of endpoints supported by llama.cpp.

        Returns:
            List of supported endpoint paths
        """
        return ["/v1/chat/completions", "/v1/completions", "/tokenize", "/health"]

    async def check_health(self, timeout: float = 5.0) -> bool:
        """
        Check if llama.cpp engine endpoint is available and responding.

        Args:
            timeout: Health check timeout in seconds

        Returns:
            True if engine is healthy, False otherwise
        """
        # First try the llama.cpp specific health endpoint
        health_url = f"{self.base_url}/health"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    health_url, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        # Try to parse JSON response
                        try:
                            health_data = await response.json()
                            status = health_data.get("status", "").lower()
                            return status == "ok" or status == "healthy"
                        except (json.JSONDecodeError, KeyError):
                            # If no valid JSON, treat 200 as healthy
                            return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

        # Fall back to parent class health check if llama.cpp specific check failed
        self.logger.debug(
            "Llama.cpp health endpoint failed, falling back to generic check"
        )
        return await super().check_health(timeout)
