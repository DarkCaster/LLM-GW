# engine/engine_client.py

from abc import ABC, abstractmethod
from typing import List
import aiohttp
import asyncio
from utils.logger import get_logger


class EngineClient(ABC):
    """
    Abstract base class for HTTP communication with LLM engines.
    """

    def __init__(self, base_url: str):
        """
        Initialize the engine client.

        Args:
            base_url: Base URL for the engine's HTTP endpoint
        """
        self.base_url = base_url.rstrip("/")
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    async def estimate_tokens(self, request_data: dict) -> int:
        """
        Calculate token requirements from incoming request.

        Args:
            request_data: The incoming request data dictionary

        Returns:
            Estimated number of tokens required for this request

        Raises:
            Exception: If token estimation fails
        """
        pass

    @abstractmethod
    def transform_request(self, request_data: dict) -> dict:
        """
        Transform request data for engine-specific format.

        Args:
            request_data: Original request data dictionary

        Returns:
            Transformed request data dictionary
        """
        pass

    @abstractmethod
    def transform_response(self, response_data: dict) -> dict:
        """
        Transform engine response to OpenAI-compatible format.

        Args:
            response_data: Engine response data dictionary

        Returns:
            Transformed response data dictionary
        """
        pass

    @abstractmethod
    def get_supported_endpoints(self) -> List[str]:
        """
        Get list of supported API endpoints for this engine.

        Returns:
            List of supported endpoint paths
        """
        pass

    async def forward_request(
        self, session: aiohttp.ClientSession, endpoint: str, request_data: dict
    ) -> aiohttp.ClientResponse:
        """
        Forward request to the engine's HTTP endpoint.

        Args:
            session: aiohttp ClientSession to use for the request
            endpoint: API endpoint path (e.g., '/v1/chat/completions')
            request_data: Request data dictionary

        Returns:
            aiohttp ClientResponse object for streaming

        Raises:
            aiohttp.ClientError: If request fails
        """
        # Transform request for this engine
        transformed_data = self.transform_request(request_data)

        # Build full URL
        url = f"{self.base_url}{endpoint}"

        self.logger.debug(f"Forwarding request to {url}")

        # Forward request to engine
        response = await session.post(
            url, json=transformed_data, headers={"Content-Type": "application/json"}
        )

        return response

    async def check_health(self, timeout: float = 5.0) -> bool:
        """
        Check if engine endpoint is available and responding.

        Args:
            timeout: Timeout in seconds for the health check

        Returns:
            True if engine is healthy and responding, False otherwise
        """
        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                # Try /health endpoint first
                try:
                    async with session.get(f"{self.base_url}/health") as response:
                        if response.status == 200:
                            self.logger.debug(
                                f"Health check passed for {self.base_url}"
                            )
                            return True
                except aiohttp.ClientError:
                    pass

                # Fallback to /v1/models endpoint
                try:
                    async with session.get(f"{self.base_url}/v1/models") as response:
                        if response.status == 200:
                            self.logger.debug(
                                f"Health check passed (via /v1/models) for {self.base_url}"
                            )
                            return True
                except aiohttp.ClientError:
                    pass

                self.logger.debug(f"Health check failed for {self.base_url}")
                return False

        except asyncio.TimeoutError:
            self.logger.debug(f"Health check timeout for {self.base_url}")
            return False
        except Exception as e:
            self.logger.debug(f"Health check error for {self.base_url}: {e}")
            return False
