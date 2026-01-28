# engine/llamacpp_engine.py

import sys
import aiohttp
import asyncio
from .engine_client import EngineClient
from .utils import parse_openai_request_content


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
        # Parse request
        try:
            content_type, prompt, max_tokens, _ = parse_openai_request_content(
                request_data
            )
        except Exception as e:
            self.logger.error(f"Error parsing request_data: {e}")
            return 1
        # if our content type is "messages" - we can construct more accurate prompt by wrapping it with chat template
        if content_type == "messages":
            # Get messages from request_data
            messages = request_data.get("messages")
            # Call /apply-template endpoint
            try:
                async with self._session.post(
                    f"{self._base_url}/apply-template",
                    json={"messages": messages},
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=60.0),
                ) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"/apply-template returned status {response.status}"
                        )
                        return max_tokens
                    template_result = await response.json()
            except Exception as e:
                self.logger.error(f"Error calling /apply-template: {e}")
                return max_tokens
            # Get prompt field from response
            prompt = template_result.get("prompt")
            if prompt is None:
                self.logger.error("No prompt field in /apply-template response")
                return max_tokens
        #TODO: here we can add support for more content types for precise estimation it needed
        else:
            self.logger.warning(
                "Trying to estimate token-count for unsupported request type or content"
            )
        # Tokenize content
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
                    return max_tokens
                tokenize_result = await response.json()
        except Exception as e:
            self.logger.error(f"Error calling /tokenize: {e}")
            return max_tokens
        # Get tokens array from response
        tokens = tokenize_result.get("tokens")
        if tokens is None or not isinstance(tokens, list):
            self.logger.error("No tokens field (or not a list) in /tokenize response")
            return max_tokens
        # Calculate total context size needed
        token_count = len(tokens)
        total_tokens = token_count + max_tokens
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
            ValueError: If endpoint is not /v1/chat/completions or /v1/embeddings
        """
        # Check that this is a chat completions request
        if path != "/v1/chat/completions" and path != "/v1/embeddings":
            self.logger.error(
                f"Unsupported URL path/endpoint for LlamaCppEngine: {path}"
            )
        # Forward the request to llama.cpp server
        full_url = f"{self._base_url}{path}"
        self.logger.debug(f"Forwarding request to {full_url}")
        # Make the request - let the response stream through
        self._request_task = asyncio.create_task(
            self._session.post(
                full_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=sys.maxsize),
            )
        )
        response = await self._request_task
        return response

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
