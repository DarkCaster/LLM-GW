# engine/engine_manager.py

import asyncio
from typing import Optional, Dict
from utils.logger import get_logger
from .engine_client import EngineClient
from .engine_process import EngineProcess
from .llamacpp_engine import LlamaCppEngine


class EngineManager:
    """
    Coordinate engine lifecycle - stop old engines, start new ones, track state.
    Singleton pattern - one manager for the entire application.
    """

    _instance: Optional["EngineManager"] = None
    _lock: asyncio.Lock = None

    def __new__(cls):
        """
        Implement singleton pattern.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the engine manager.
        """
        if self._initialized:
            return

        self.logger = get_logger(self.__class__.__name__)

        # Current engine state
        self.current_engine_process: Optional[EngineProcess] = None
        self.current_engine_client: Optional[EngineClient] = None
        self.current_variant_config: Optional[Dict] = None
        self.current_model_name: Optional[str] = None

        # Lock for preventing concurrent engine switches
        if EngineManager._lock is None:
            EngineManager._lock = asyncio.Lock()

        self._initialized = True

        self.logger.info("EngineManager initialized")

    async def ensure_engine(
        self, model_name: str, variant_config: Dict, engine_type: str
    ) -> EngineClient:
        """
        Ensure the correct engine is running for the given model and variant.
        If the same variant is already running, return current client.
        If different variant needed, switch engines.

        Args:
            model_name: Name of the model
            variant_config: Configuration dictionary for the variant
            engine_type: Type of engine (e.g., "llama.cpp")

        Returns:
            EngineClient instance for the running engine

        Raises:
            Exception: If engine fails to start or becomes unhealthy
        """
        async with EngineManager._lock:
            # Check if the same variant is already running
            if self._is_same_variant(model_name, variant_config):
                # Verify health of current engine
                if self.current_engine_client:
                    is_healthy = await self.current_engine_client.check_health(
                        timeout=5.0
                    )
                    if is_healthy:
                        self.logger.info(
                            f"Engine already running with correct variant for model '{model_name}'"
                        )
                        return self.current_engine_client
                    else:
                        self.logger.warning("Current engine is unhealthy, restarting")

            # Need to switch engines
            self.logger.info(f"Switching to variant for model '{model_name}'")

            # Stop current engine
            await self._stop_current_engine()

            # Start new engine
            await self._start_new_engine(model_name, variant_config, engine_type)

            return self.current_engine_client

    def _is_same_variant(self, model_name: str, variant_config: Dict) -> bool:
        """
        Check if the requested variant is the same as currently running.

        Args:
            model_name: Name of the model
            variant_config: Configuration dictionary for the variant

        Returns:
            True if same variant is running, False otherwise
        """
        if self.current_model_name != model_name:
            return False

        if self.current_variant_config is None:
            return False

        # Compare key fields to determine if it's the same variant
        # We'll compare binary, connect URL, and context size
        current_binary = self.current_variant_config.get("binary")
        current_connect = self.current_variant_config.get("connect")
        current_context = self.current_variant_config.get("context")

        new_binary = variant_config.get("binary")
        new_connect = variant_config.get("connect")
        new_context = variant_config.get("context")

        return (
            current_binary == new_binary
            and current_connect == new_connect
            and current_context == new_context
        )

    async def _stop_current_engine(self) -> None:
        """
        Stop the currently running engine if any.
        """
        if self.current_engine_process is None:
            self.logger.debug("No current engine to stop")
            return

        self.logger.info(
            f"Stopping current engine for model '{self.current_model_name}'"
        )

        try:
            await self.current_engine_process.stop(timeout=15.0)
            self.logger.info("Engine stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping engine: {e}")
        finally:
            # Clear current state
            self.current_engine_process = None
            self.current_engine_client = None
            self.current_variant_config = None
            self.current_model_name = None

    async def _start_new_engine(
        self, model_name: str, variant_config: Dict, engine_type: str
    ) -> None:
        """
        Start a new engine with the given configuration.

        Args:
            model_name: Name of the model
            variant_config: Configuration dictionary for the variant
            engine_type: Type of engine (e.g., "llama.cpp")

        Raises:
            ValueError: If engine type is unknown
            Exception: If engine fails to start or doesn't become ready
        """
        # Extract configuration
        binary = variant_config.get("binary")
        args = variant_config.get("args", [])
        connect_url = variant_config.get("connect")

        if not binary:
            raise ValueError("Variant configuration missing 'binary' field")
        if not connect_url:
            raise ValueError("Variant configuration missing 'connect' field")

        self.logger.info(
            f"Starting new engine for model '{model_name}' with engine type '{engine_type}'"
        )

        # Create appropriate EngineClient based on engine type
        if engine_type == "llama.cpp":
            engine_client = LlamaCppEngine(connect_url)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        # Create EngineProcess
        engine_process = EngineProcess(binary, args, work_dir=None)

        # Start the process
        try:
            await engine_process.start()
        except Exception as e:
            self.logger.error(f"Failed to start engine process: {e}")
            raise

        # Wait for engine to become ready
        try:
            await self._wait_for_engine_ready(engine_client, timeout=60.0)
        except Exception as e:
            # Engine failed to become ready, stop the process
            self.logger.error(f"Engine failed to become ready: {e}")
            await engine_process.stop(timeout=10.0)
            raise

        # Store current engine state
        self.current_engine_process = engine_process
        self.current_engine_client = engine_client
        self.current_variant_config = variant_config
        self.current_model_name = model_name

        self.logger.info(f"Engine started successfully for model '{model_name}'")

    async def _wait_for_engine_ready(
        self, engine_client: EngineClient, timeout: float
    ) -> bool:
        """
        Wait for the engine HTTP endpoint to become available.

        Args:
            engine_client: EngineClient instance to check
            timeout: Maximum time to wait in seconds

        Returns:
            True if engine becomes ready

        Raises:
            TimeoutError: If engine doesn't become ready within timeout
        """
        self.logger.info("Waiting for engine to become ready...")

        start_time = asyncio.get_event_loop().time()
        check_interval = 1.0  # Check every 1 second

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed >= timeout:
                raise TimeoutError(
                    f"Engine did not become ready within {timeout} seconds"
                )

            # Check health
            is_ready = await engine_client.check_health(timeout=5.0)

            if is_ready:
                self.logger.info(f"Engine is ready (took {elapsed:.1f}s)")
                return True

            # Log progress
            self.logger.debug(f"Waiting for engine to be ready... {elapsed:.1f}s")

            # Wait before next check
            await asyncio.sleep(check_interval)

    def get_current_client(self) -> Optional[EngineClient]:
        """
        Get the currently running engine client.

        Returns:
            Current EngineClient instance if available, None otherwise
        """
        return self.current_engine_client

    async def shutdown(self) -> None:
        """
        Shutdown the engine manager and stop any running engines.
        """
        self.logger.info("Shutting down EngineManager")

        async with EngineManager._lock:
            await self._stop_current_engine()

        self.logger.info("EngineManager shutdown complete")
