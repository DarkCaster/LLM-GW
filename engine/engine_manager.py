"""
EngineManager - Coordinates engine lifecycle and state management.

Manages engine processes, handles engine switching, and ensures correct
engine is running for incoming requests.
"""

import asyncio
from typing import Dict, Optional, Any

from .engine_client import EngineClient
from .llamacpp_engine import LlamaCppEngine
from .engine_process import EngineProcess
from utils.logger import get_logger


class EngineManager:
    """
    Manages LLM engine lifecycle and state.

    Singleton pattern - one manager instance for the entire application.
    """

    _instance: Optional["EngineManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize EngineManager (singleton)."""
        if not self._initialized:
            self._logger = get_logger(self.__class__.__name__)

            # Current engine state
            self.current_engine_process: Optional[EngineProcess] = None
            self.current_engine_client: Optional[EngineClient] = None
            self.current_variant_config: Optional[Dict[str, Any]] = None
            self.current_model_name: Optional[str] = None

            # Lock for preventing concurrent engine switches
            self._lock = asyncio.Lock()

            # Engine type mapping
            self._engine_types = {
                "llama.cpp": LlamaCppEngine,
                # Future: add other engine types here
            }

            self._initialized = True
            self._logger.info("EngineManager initialized")

    async def ensure_engine(
        self, model_name: str, variant_config: Dict[str, Any], engine_type: str
    ) -> EngineClient:
        """
        Ensure the correct engine is running for the requested model and variant.

        Args:
            model_name: Name of the model to load
            variant_config: Configuration for the engine variant
            engine_type: Type of engine (e.g., "llama.cpp")

        Returns:
            EngineClient instance for communicating with the engine

        Raises:
            ValueError: If engine_type is not supported
            RuntimeError: If engine fails to start or health check fails
        """
        async with self._lock:
            # Check if we already have the requested engine running
            if self._is_same_engine(model_name, variant_config):
                self._logger.debug(
                    f"Requested engine already running: {model_name}, "
                    f"context={variant_config.get('context', 'N/A')}"
                )

                # Verify engine is still healthy
                try:
                    if await self.current_engine_client.check_health():
                        return self.current_engine_client
                    else:
                        self._logger.warning(
                            "Current engine failed health check, restarting..."
                        )
                except Exception as e:
                    self._logger.error(
                        f"Health check failed: {e}, restarting engine..."
                    )

                # If health check failed, continue with engine switch

            self._logger.info(
                f"Switching engine: {model_name}, "
                f"context={variant_config.get('context', 'N/A')}, "
                f"type={engine_type}"
            )

            # Stop current engine if running
            await self._stop_current_engine()

            # Start new engine
            await self._start_new_engine(model_name, variant_config, engine_type)

            return self.current_engine_client

    async def _stop_current_engine(self) -> None:
        """Stop the currently running engine process."""
        if not self.current_engine_process:
            self._logger.debug("No engine running to stop")
            return

        try:
            await self.current_engine_process.stop(timeout=15.0)
            self._logger.info(f"Stopped engine for model: {self.current_model_name}")
        except Exception as e:
            self._logger.error(f"Error stopping engine: {e}")
        finally:
            self.current_engine_process = None
            self.current_engine_client = None
            self.current_variant_config = None
            self.current_model_name = None

    async def _start_new_engine(
        self, model_name: str, variant_config: Dict[str, Any], engine_type: str
    ) -> None:
        """
        Start a new engine with the given configuration.

        Args:
            model_name: Name of the model to load
            variant_config: Configuration for the engine variant
            engine_type: Type of engine (e.g., "llama.cpp")

        Raises:
            ValueError: If engine_type is not supported or config is invalid
            RuntimeError: If engine fails to start or health check times out
        """
        if engine_type not in self._engine_types:
            raise ValueError(f"Unsupported engine type: {engine_type}")

        # Validate required configuration fields
        required_fields = ["binary", "args", "connect"]
        for field in required_fields:
            if field not in variant_config:
                raise ValueError(f"Missing required field in variant config: {field}")

        # Create engine client
        engine_client_class = self._engine_types[engine_type]
        base_url = variant_config["connect"]
        self.current_engine_client = engine_client_class(base_url=base_url)

        # Create and start engine process
        binary_path = variant_config["binary"]
        args = variant_config["args"]

        self.current_engine_process = EngineProcess(
            binary_path=binary_path, args=args, logger=self._logger
        )

        try:
            await self.current_engine_process.start()
        except Exception as e:
            self._logger.error(f"Failed to start engine process: {e}")
            # Clean up on failure
            self.current_engine_process = None
            self.current_engine_client = None
            raise RuntimeError(f"Failed to start engine: {e}")

        # Wait for engine to be ready
        if not await self._wait_for_engine_ready():
            # Clean up on timeout
            try:
                await self._stop_current_engine()
            except Exception as e:
                self._logger.error(f"Error cleaning up failed engine: {e}")
            raise RuntimeError("Engine failed to become ready within timeout")

        # Store state
        self.current_variant_config = variant_config.copy()
        self.current_model_name = model_name

        self._logger.info(
            f"Engine started successfully: {model_name} "
            f"(PID: {self.current_engine_process.get_pid()})"
        )

    async def _wait_for_engine_ready(
        self, timeout: float = 60.0, check_interval: float = 1.0
    ) -> bool:
        """
        Wait for engine to become ready by polling health checks.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Time between health checks in seconds

        Returns:
            True if engine becomes ready within timeout, False otherwise
        """
        if not self.current_engine_client:
            return False

        start_time = asyncio.get_event_loop().time()
        elapsed = 0.0

        self._logger.info(f"Waiting for engine to be ready (timeout: {timeout}s)...")

        while elapsed < timeout:
            try:
                if await self.current_engine_client.check_health(timeout=5.0):
                    self._logger.info(f"Engine ready after {elapsed:.1f}s")
                    return True
            except Exception as e:
                self._logger.debug(f"Health check attempt failed: {e}")

            # Wait before next check
            await asyncio.sleep(check_interval)
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed % 5.0 < check_interval:  # Log every ~5 seconds
                self._logger.debug(
                    f"Still waiting for engine... {elapsed:.1f}s elapsed"
                )

        self._logger.error(f"Engine failed to become ready within {timeout}s timeout")
        return False

    def _is_same_engine(self, model_name: str, variant_config: Dict[str, Any]) -> bool:
        """
        Check if requested engine matches currently running engine.

        Args:
            model_name: Requested model name
            variant_config: Requested variant configuration

        Returns:
            True if same engine is already running, False otherwise
        """
        if not self.current_model_name or not self.current_variant_config:
            return False

        # Check model name
        if self.current_model_name != model_name:
            return False

        # Compare key configuration parameters
        current_ctx = self.current_variant_config.get("context")
        requested_ctx = variant_config.get("context")

        if current_ctx != requested_ctx:
            return False

        # Check if binary and args are the same (for safety)
        current_binary = self.current_variant_config.get("binary")
        requested_binary = variant_config.get("binary")

        if current_binary != requested_binary:
            return False

        # We could compare more fields, but context and binary are the main ones

        return True

    def get_current_client(self) -> Optional[EngineClient]:
        """
        Get the client for the currently running engine.

        Returns:
            EngineClient if engine is running, None otherwise
        """
        return self.current_engine_client

    async def shutdown(self) -> None:
        """Shutdown the engine manager and stop any running engine."""
        self._logger.info("Shutting down EngineManager...")
        async with self._lock:
            await self._stop_current_engine()
        self._logger.info("EngineManager shutdown complete")

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current engine state for monitoring/debugging.

        Returns:
            Dictionary with current engine state information
        """
        state = {
            "model_name": self.current_model_name,
            "engine_running": self.current_engine_process is not None,
            "variant_config": self.current_variant_config,
        }

        if self.current_engine_process:
            state.update(
                {
                    "pid": self.current_engine_process.get_pid(),
                    "status": self.current_engine_process.get_status(),
                    "uptime": self.current_engine_process.get_uptime(),
                }
            )

        return state
