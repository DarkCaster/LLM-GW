# engine/llamacpp_engine.py

import aiohttp
import asyncio
import random
from .engine_client import EngineClient


class LlamaCppEngine(EngineClient):
    """
    Concrete implementation of EngineClient for llama.cpp engines.
    """

    def __init__(self, session: aiohttp.ClientSession, base_url: str):
        """
        Initialize LlamaCppEngine with base URL.

        Args:
            base_url: Base URL for the llama.cpp server (e.g., "http://127.0.0.1:8080")
        """
        super().__init__()
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.logger.info(f"Initialized LlamaCppEngine with base_url: {self.base_url}")

    async def estimate_tokens(self, request_data: dict) -> int:
        """
        Calculate token requirements from incoming request.

        For now, this is a stub that returns a random value between 1 and 512.
        Will be properly implemented later.

        Args:
            request_data: Dictionary containing the request data

        Returns:
            Estimated number of tokens (stub: random value 1-512)
        """
        # Stub implementation - will be addressed later
        return random.randint(1, 512)

    async def forward_request(
        self, url: str, request_data: dict
    ) -> aiohttp.ClientResponse:
        """
        Forward request to llama.cpp server endpoint.

        Only supports /v1/chat/completions endpoint for now. Handles both normal
        and streaming responses.

        Args:
            url: URL of the request to forwarding, without domain and protocol, example: /v1/chat/completions
            request_data: Dictionary containing the request data

        Returns:
            aiohttp ClientResponse object

        Raises:
            ValueError: If endpoint is not /v1/chat/completions
        """
        # Check that this is a chat completions request
        if url != "/v1/chat/completions":
            raise ValueError(
                f"LlamaCppEngine only supports /v1/chat/completions endpoint for now, got: {url}"
            )

        # Transform request data
        transformed_data = self._transform_request(request_data)

        # Forward the request to llama.cpp server
        full_url = f"{self.base_url}/v1/chat/completions"

        self.logger.debug(f"Forwarding request to {full_url}")

        # Make the request - let the response stream through
        response = await self.session.post(
            full_url,
            json=transformed_data,
            headers={"Content-Type": "application/json"},
        )

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

    async def check_health(self) -> bool:
        """
        Check llama.cpp engine health using /health endpoint.

        Returns:
            True if engine is healthy, False otherwise
        """
        health_url = f"{self.base_url}/health"

        try:
            async with self.session.get(
                health_url, timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                if response.status == 200:
                    self.logger.debug(f"Health check passed for {self.base_url}")
                    return True
                else:
                    self.logger.debug(
                        f"Health check failed with status {response.status} for {self.base_url}"
                    )
                    return False
        except asyncio.TimeoutError:
            self.logger.warning(f"Health check timeout for {self.base_url}")
            return False
        except aiohttp.ClientError as e:
            self.logger.warning(
                f"Health check connection error for {self.base_url}: {e}"
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Unexpected error during health check for {self.base_url}: {e}"
            )
            return False
