# engine/standalone_tokenizer.py

from abc import ABC, abstractmethod
import logger


class StandaloneTokenizer(ABC):
    """
    Abstract base class for standalone tokenizer implementations.
    Used for estimating token counts without requiring a running engine.
    """

    def __init__(self):
        """Initialize the standalone tokenizer."""
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
