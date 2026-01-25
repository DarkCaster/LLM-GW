# engine/engine_client.py

from abc import ABC, abstractmethod
import logger
import aiohttp


class EngineClient(ABC):
    """
    Abstract base class for HTTP communication with LLM engines.
    """

    def __init__(self):
        """Initialize the engine client."""
        self.logger = logger.get_logger(self.__class__.__name__)

    @abstractmethod
    async def estimate_tokens(self, request_data: dict) -> int:
        """
        Calculate token requirements from incoming request.

        Args:
            request_data: Dictionary containing the request data from await request.json()

        Returns:
            Estimated number of tokens required for the request
        """
        pass

    @abstractmethod
    async def forward_request(
        self, path: str, request_data: dict
    ) -> aiohttp.ClientResponse:
        """
        Process and forward request to the engine.

        This method may transform the request before forwarding it to the engine,
        and may also transform the response before returning.

        Args:
            path: URL path/endpoint of the request to forwarding, example: /v1/chat/completions
            request_data: Dictionary containing the request data

        Returns:
            aiohttp ClientResponse object
        """
        pass

    @abstractmethod
    def terminate_request(self) -> None:
        """
        Terminate currently running request
        """
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        """
        Check engine health and return True if engine is up and running.

        Returns:
            True if engine is healthy, False otherwise
        """
        pass
