# engine/llamacpp_engine.py

import aiohttp
import asyncio
from .engine_client import EngineClient


class LlamaCppEngineClient(EngineClient):
    """
    Concrete implementation of EngineClient for llama.cpp engines.
    """

    def __init__(
        self, session: aiohttp.ClientSession, base_url: str, health_check_timeout: float
    ):
        """
        Initialize LlamaCppEngine with base URL.

        Args:
            base_url: Base URL for the llama.cpp server (e.g., "http://127.0.0.1:8080")
        """
        super().__init__()
        self._health_check_timeout = health_check_timeout
        self._session = session
        self._base_url = base_url.rstrip("/")
        self._request_task = None
        self.logger.debug(f"Initialized LlamaCppEngine with base_url: {self._base_url}")

    async def estimate_tokens(self, request_data: dict) -> int:
        """
        Tokenize chat messages and get token requirements from incoming request (including safe margin)

        Args:
            request_data: Dictionary containing the request data

        Returns:
            Estimated number of context size tokens needed to process request
        """
        # Get max_tokens field from request_data
        max_tokens = request_data.get("max_tokens")
        if max_tokens is None:
            max_tokens = request_data.get("max_completion_tokens")
        if max_tokens is None:
            self.logger.warning(
                "No max_tokens or max_completion_tokens in request, defaulting to 512"
            )
            max_tokens = 512
        # Get messages from request_data
        messages = request_data.get("messages")
        if messages is None:
            self.logger.error("No messages field in request_data")
            return max_tokens + 32
        # Call /apply-template endpoint
        apply_template_url = f"{self._base_url}/apply-template"
        try:
            async with self._session.post(
                apply_template_url,
                json={"messages": messages},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60.0),
            ) as response:
                if response.status != 200:
                    self.logger.error(
                        f"/apply-template returned status {response.status}"
                    )
                    return max_tokens + 32
                template_result = await response.json()
        except Exception as e:
            self.logger.error(f"Error calling /apply-template: {e}")
            return max_tokens + 32
        # Get prompt field from response
        prompt = template_result.get("prompt")
        if prompt is None:
            self.logger.error("No prompt field in /apply-template response")
            return max_tokens + 32
        # Call /tokenize endpoint
        tokenize_url = f"{self._base_url}/tokenize"
        try:
            async with self._session.post(
                tokenize_url,
                json={"content": prompt},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60.0),
            ) as response:
                if response.status != 200:
                    self.logger.error(f"/tokenize returned status {response.status}")
                    return max_tokens + 32
                tokenize_result = await response.json()
        except Exception as e:
            self.logger.error(f"Error calling /tokenize: {e}")
            return max_tokens + 32
        # Get tokens array from response
        tokens = tokenize_result.get("tokens")
        if tokens is None or not isinstance(tokens, list):
            self.logger.error("No tokens field (or not a list) in /tokenize response")
            return max_tokens + 32
        # Calculate total context size needed
        token_count = len(tokens)
        total_tokens = token_count + max_tokens + 32
        self.logger.debug(
            f"Token estimation: prompt={token_count}, max_tokens={max_tokens}, total={total_tokens}"
        )
        return total_tokens

    async def forward_request(
        self, path: str, request_data: dict
    ) -> aiohttp.ClientResponse:
        """
        Forward request to llama.cpp server endpoint.

        Only supports /v1/chat/completions endpoint for now. Handles both normal
        and streaming responses.

        Args:
            path: URL path/endpoint of the request to forwarding, example: /v1/chat/completions
            request_data: Dictionary containing the request data

        Returns:
            aiohttp ClientResponse object, do not forget to release it manually after use

        Raises:
            ValueError: If endpoint is not /v1/chat/completions
        """
        # Check that this is a chat completions request
        if path != "/v1/chat/completions":
            self.logger.error(
                f"Unsupported URL path/endpoint for LlamaCppEngine: {path}"
            )
        # Transform request data
        transformed_data = self._transform_request(request_data)
        # Forward the request to llama.cpp server
        full_url = f"{self._base_url}{path}"
        self.logger.debug(f"Forwarding request to {full_url}")
        # Make the request - let the response stream through
        self._request_task = asyncio.create_task(
            self._session.post(
                full_url,
                json=transformed_data,
                headers={"Content-Type": "application/json"},
            )
        )
        response = await self._request_task
        return response

    def _transform_request(self, request_data: dict) -> dict:
        """
        Transform request data before forwarding to llama.cpp.

        For now, this is a stub that returns the original request.
        Will be properly implemented later to handle llama.cpp-specific transformations if needed.

        Args:
            request_data: Original request data

        Returns:
            Transformed request data (stub: returns original)
        """
        # Stub implementation - will be addressed later
        return request_data

    def terminate_request(self) -> None:
        """
        Terminate currently running request
        """
        if self._request_task is not None:
            self._request_task.cancel()

    async def check_health(self) -> bool:
        """
        Check llama.cpp engine health using /health endpoint.

        Returns:
            True if engine is healthy, False otherwise
        """
        health_url = f"{self._base_url}/health"
        try:
            async with self._session.get(
                health_url,
                timeout=aiohttp.ClientTimeout(total=self._health_check_timeout),
            ) as response:
                if response.status == 200:
                    self.logger.debug(f"Health check passed for {self._base_url}")
                    return True
                else:
                    self.logger.debug(
                        f"Health check failed with status {response.status} for {self._base_url}"
                    )
                    return False
        except asyncio.TimeoutError:
            self.logger.warning(f"Health check timeout for {self._base_url}")
            return False
        except aiohttp.ClientError as e:
            self.logger.warning(
                f"Health check connection error for {self._base_url}: {e}"
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Unexpected error during health check for {self._base_url}: {e}"
            )
            return False
