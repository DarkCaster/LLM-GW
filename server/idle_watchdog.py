# server/idle_watchdog.py

import asyncio
from typing import Callable, Awaitable
from utils.logger import get_logger


class IdleWatchdog:
    """
    Run specified function on timeout
    """

    def __init__(self):
        """Initialize IdleWatchdog."""
        self.logger = get_logger(self.__class__.__name__)
        self._timer_task: asyncio.Task | None = None
        self._timeout: float = 0.0
        self._callback: Callable[[], Awaitable[None]] | None = None

    def disarm(self) -> None:
        """Stop current timer."""
        if self._timer_task is not None and not self._timer_task.done():
            self.logger.debug("Disarming idle watchdog timer")
            self._timer_task.cancel()
            self._timer_task = None

    def rearm(self, timeout: float, callback: Callable[[], Awaitable[None]]) -> None:
        """
        Rearm the watchdog timer with a new timeout and callback.

        Args:
            timeout: Timeout in seconds
            callback: Async function to call when timeout triggers
        """
        # Disarm any existing timer
        self.disarm()

        # Store timeout and callback
        self._timeout = timeout
        self._callback = callback

        # Start new timer
        if timeout > 0:
            self.logger.debug(f"Arming idle watchdog timer with timeout {timeout}s")
            self._timer_task = asyncio.create_task(self._timer_handler())
        else:
            self.logger.debug("Timeout is 0 or negative, not arming timer")

    async def _timer_handler(self) -> None:
        """Internal timer handler that waits for timeout and calls callback."""
        try:
            await asyncio.sleep(self._timeout)
            self.logger.info(f"Idle timeout triggered after {self._timeout}s")
            if self._callback is not None:
                await self._callback()
        except asyncio.CancelledError:
            self.logger.debug("Timer cancelled")
        except Exception as e:
            self.logger.error(f"Error in idle watchdog callback: {e}", exc_info=True)
