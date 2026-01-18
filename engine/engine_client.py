"""
Base EngineClient abstract class for LLM engine communication.

Provides abstract interface for engine-specific implementations.
"""

import abc
import asyncio
import logging
from typing import List, Optional

import aiohttp


class EngineClient(abc.ABC):
    """Abstract base class for HTTP communication with LLM engines."""

    def __init__(self, base_url: str, logger: Optional[logging.Logger] = None):
        """
        Initialize EngineClient with base URL.

        Args:
            base_url: Base URL for engine HTTP endpoint
            logger: Logger instance (uses class name if not provided)
        """
        self.base_url = base_url.rstrip("/")
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        return self._logger

    @abc.abstractmethod
    async def estimate_tokens(self, request_data: dict) -> int:
        """
        Calculate token requirements from incoming request.

        Args:
            request_data: Request data dictionary

        Returns:
            Estimated token count

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement estimate_tokens")

    @abc.abstractmethod
    def transform_request(self, request_data: dict) -> dict:
        """
        Transform request for engine-specific format.

        Args:
            request_data: Original request data

        Returns:
            Transformed request data
        """
        raise NotImplementedError("Subclasses must implement transform_request")

    @abc.abstractmethod
    def transform_response(self, response_data: dict) -> dict:
        """
        Transform response to OpenAI-compatible format.

        Args:
            response_data: Engine response data

        Returns:
            Transformed response data
        """
        raise NotImplementedError("Subclasses must implement transform_response")

    @abc.abstractmethod
    def get_supported_endpoints(self) -> List[str]:
        """
        Get list of supported API endpoints.

        Returns:
            List of endpoint paths
        """
        raise NotImplementedError("Subclasses must implement get_supported_endpoints")

    async def forward_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        request_data: dict,
        timeout: float = 60.0,
    ) -> aiohttp.ClientResponse:
        """
        Forward request to engine HTTP endpoint.

        Args:
            session: aiohttp ClientSession
            endpoint: API endpoint path (e.g., "/v1/chat/completions")
            request_data: Request data to forward
            timeout: Request timeout in seconds

        Returns:
            aiohttp ClientResponse object

        Raises:
            ValueError: If endpoint is not supported
            aiohttp.ClientError: On HTTP request failure
        """
        if endpoint not in self.get_supported_endpoints():
            raise ValueError(f"Endpoint {endpoint} is not supported by this engine")

        transformed_data = self.transform_request(request_data)
        url = f"{self.base_url}{endpoint}"

        self.logger.debug(f"Forwarding request to {url}")

        try:
            return await session.post(
                url,
                json=transformed_data,
                timeout=aiohttp.ClientTimeout(total=timeout),
            )
        except aiohttp.ClientError as e:
            self.logger.error(f"Failed to forward request to {url}: {e}")
            raise

    async def check_health(self, timeout: float = 5.0) -> bool:
        """
        Check if engine endpoint is available and responding.

        Args:
            timeout: Health check timeout in seconds

        Returns:
            True if engine is healthy, False otherwise
        """
        # Try common health endpoints
        health_endpoints = ["/health", "/v1/models", "/"]

        async with aiohttp.ClientSession() as session:
            for endpoint in health_endpoints:
                url = f"{self.base_url}{endpoint}"
                try:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        if response.status < 500:  # 2xx, 3xx, 4xx are okay
                            self.logger.debug(f"Health check passed for {url}")
                            return True
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    continue  # Try next endpoint

        self.logger.warning(f"All health checks failed for {self.base_url}")
        return False
