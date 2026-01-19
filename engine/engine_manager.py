# engine/engine_manager.py

import aiohttp
import asyncio
import python_lua_helper
import sys
from utils.logger import get_logger
from typing import Optional, Dict, Any
from .engine_client import EngineClient
from .llamacpp_engine import LlamaCppEngine
from .engine_process import EngineProcess


class EngineManager:
    """
    Coordinate engine lifecycle - stop old engines, start new ones, track state.
    Use one instance for per application.
    """

    def __init__(
        self, session: aiohttp.ClientSession, cfg: python_lua_helper.PyLuaHelper
    ):
        """
        Initialize EngineManager.

        Args:
            session: aiohttp.ClientSession, session used
            cfg: PyLuaHelper configuration object
        """
        # Only initialize once
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.logger = get_logger(self.__class__.__name__)
        self.session = session
        self.cfg = cfg

        # Engine state
        self._current_engine_process: Optional[EngineProcess] = None
        self._current_engine_client: Optional[EngineClient] = None
        self._current_model_name: Optional[str] = None
        self._current_config: Optional[Dict[str, Any]] = None
        self._current_engine_type: Optional[str] = None

        # Lock to prevent concurrent engine switches
        self._lock = asyncio.Lock()

        self.logger.info("EngineManager initialized")

    @property
    def current_engine_client(self) -> Optional[EngineClient]:
        """
        Get the client for the current engine (if running).

        Returns:
            Current EngineClient instance or None if no engine is running
        """
        return self._current_engine_client

    def check_model_configuration(self, model_name: str, required_config: dict) -> bool:
        """
        Check that currently running model is suitable for selected configuration.

        Args:
            model_name: Name of the model to check
            required_config: Configuration dictionary (contents vary by engine type)

        Returns:
            True if engine for requested model is already loaded with suitable config
        """
        # No engine running
        if self._current_engine_client is None:
            return False

        # Different model
        if self._current_model_name != model_name:
            return False

        # Check engine-specific configuration
        if self._current_engine_type == "llama.cpp":
            return self._check_llamacpp_config(required_config)

        return False

    def _check_llamacpp_config(self, required_config: dict) -> bool:
        """
        Check if current llamacpp engine configuration matches requirements.

        Args:
            required_config: Required configuration dictionary

        Returns:
            True if current engine is llamacpp type (stub implementation)
        """
        if not self._current_config or not required_config:
            return False
        # check cases when we requested config context_estimation
        if required_config.get("operation", "unknown") == "context_estimation":
            if self._current_config.get("operation", "unknown") == "context_estimation":
                # currently loaded model was already configured specifically for context_estimation
                return True
            else:
                # check can we actually use currently loaded model configuration for context estimation
                # get variant index from current configuration (return false if no index)
                variant_index = self._current_config.get("variant_index")
                if variant_index is None:
                    return False
                # get model index from cfg for current model name, return false if we cannot detect it (log internal error)
                model_index = None
                for i in self.cfg.get_table_seq("models"):
                    if self.cfg.get(f"models.{i}.name") == self._current_model_name:
                        model_index = i
                        break
                if model_index is None:
                    self.logger.error(
                        f"Internal error: Model '{self._current_model_name}' not found in configuration"
                    )
                    return False
                # get `tokenize` value from config for model with known index and variant index, return false if tokenize is false
                if not self.cfg.get_bool(
                    f"models.{model_index}.variants.{variant_index}.tokenize", False
                ):
                    return False
                # currently loaded model is suitable
                return True
        elif required_config.get("operation", "unknown") == "text_query":
            if self._current_config.get("operation", "unknown") == "text_query":
                # check can we actually use currently loaded model configuration for text query
                context_required = required_config.get(
                    "context_size_required", sys.maxsize
                )
                # get variant index from current configuration (return false if no index)
                variant_index = self._current_config.get("variant_index")
                if variant_index is None:
                    return False
                # get model index from cfg for current model name, return false if we cannot detect it (log internal error)
                model_index = None
                for i in self.cfg.get_table_seq("models"):
                    if self.cfg.get(f"models.{i}.name") == self._current_model_name:
                        model_index = i
                        break
                if model_index is None:
                    self.logger.error(
                        f"Internal error: Model '{self._current_model_name}' not found in configuration"
                    )
                    return False
                # get `context` value from cfg (with fallback 0), compare with context_required return false if context value is smaller
                current_context = self.cfg.get_int(
                    f"models.{model_index}.variants.{variant_index}.context", 0
                )
                if current_context < context_required:
                    return False
                # currently loaded model is suitable
                return True

        # For any other case - currently loaded model is not suitable
        return False

    async def ensure_engine(
        self, model_name: str, required_config: dict, engine_type: str
    ) -> EngineClient:
        """
        Ensure the correct engine is running with the required configuration.

        Args:
            model_name: Name of the model to load
            required_config: Configuration dictionary with variant_index
            engine_type: Type of engine ("llama.cpp", etc.)

        Returns:
            EngineClient instance for the running engine

        Raises:
            ValueError: If model not found, engine type not supported, or config invalid
            TimeoutError: If engine fails to become ready
        """
        async with self._lock:
            self.logger.info(
                f"Ensuring engine for model '{model_name}' with engine type '{engine_type}'"
            )

            # Find model in configuration
            model_index = None
            for i in self.cfg.get_table_seq("models"):
                if self.cfg.get(f"models.{i}.name") == model_name:
                    model_index = i
                    break

            if model_index is None:
                raise ValueError(f"Model '{model_name}' not found in configuration")

            # Get engine type for the model
            cfg_engine_type = self.cfg.get(f"models.{model_index}.engine")

            # Only support llama.cpp for now
            if cfg_engine_type != "llama.cpp":
                raise ValueError(
                    f"Engine type '{cfg_engine_type}' not supported yet. "
                    "Only 'llama.cpp' is currently supported."
                )

            if engine_type != "llama.cpp":
                raise ValueError(
                    f"Requested engine type '{engine_type}' not supported yet. "
                    "Only 'llama.cpp' is currently supported."
                )

            # Check if current engine configuration is suitable
            if self.check_model_configuration(model_name, required_config):
                # Verify health
                if self._current_engine_client is not None:
                    if await self._current_engine_client.check_health():
                        self.logger.info(
                            f"Current engine for model '{model_name}' is already running and healthy"
                        )
                        return self._current_engine_client
                    else:
                        self.logger.warning(
                            f"Current engine for model '{model_name}' failed health check"
                        )

            # Need to start or restart engine
            self.logger.info(f"Starting new engine for model '{model_name}'")
            await self._stop_current_engine()
            await self._start_new_engine(model_name, required_config, engine_type)

            return self._current_engine_client

    async def _stop_current_engine(self) -> None:
        """
        Stop the currently running engine.
        """
        if self._current_engine_process is None:
            self.logger.debug("No current engine to stop")
            return

        self.logger.info(
            f"Stopping current engine for model '{self._current_model_name}'"
        )

        try:
            await self._current_engine_process.stop(timeout=15.0)
        except Exception as e:
            self.logger.error(f"Error stopping engine: {e}")

        # Clear current state
        self._current_engine_process = None
        self._current_engine_client = None
        self._current_model_name = None
        self._current_config = None
        self._current_engine_type = None

        self.logger.info("Current engine stopped and state cleared")

    async def _start_new_engine(
        self, model_name: str, required_config: dict, engine_type: str
    ) -> None:
        """
        Start a new engine with the specified configuration.

        Args:
            model_name: Name of the model to load
            required_config: Configuration dictionary with variant_index
            engine_type: Type of engine ("llama.cpp", etc.)

        Raises:
            ValueError: If engine type not supported or configuration invalid
            TimeoutError: If engine fails to become ready
        """
        # Only support llama.cpp for now
        if engine_type != "llama.cpp":
            raise ValueError(
                f"Engine type '{engine_type}' not supported yet. "
                "Only 'llama.cpp' is currently supported."
            )

        # Find model in configuration
        model_index = None
        for i in self.cfg.get_table_seq("models"):
            if self.cfg.get(f"models.{i}.name") == model_name:
                model_index = i
                break

        if model_index is None:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        # Extract variant_index from required_config
        variant_index = required_config.get("variant_index")
        if variant_index is None:
            raise ValueError("variant_index not specified in required_config")

        # Get variant configuration
        variant_key = f"models.{model_index}.variants.{variant_index}"

        # Extract binary, args, and connect URL
        binary = self.cfg.get(f"{variant_key}.binary")
        if not binary:
            raise ValueError(f"Binary path not found for variant {variant_index}")

        connect_url = self.cfg.get(f"{variant_key}.connect")
        if not connect_url:
            raise ValueError(f"Connect URL not found for variant {variant_index}")

        # Get args as list
        args = self.cfg.get_list(f"{variant_key}.args")

        self.logger.info(
            f"Starting engine: binary={binary}, connect={connect_url}, "
            f"args count={len(args)}"
        )

        # Create EngineClient based on engine type
        if engine_type == "llama.cpp":
            engine_client = LlamaCppEngine(self.session, connect_url)
        else:
            raise ValueError(f"Engine type '{engine_type}' not implemented")

        # Create and start EngineProcess
        engine_process = EngineProcess(binary, args)
        await engine_process.start()

        # Wait for engine to become ready
        try:
            await self._wait_for_engine_ready(engine_client, timeout=60.0)
        except Exception as e:
            # If engine fails to become ready, stop the process
            self.logger.error(f"Engine failed to become ready: {e}")
            await engine_process.stop()
            raise

        # Store engine state
        self._current_engine_process = engine_process
        self._current_engine_client = engine_client
        self._current_model_name = model_name
        self._current_config = required_config
        self._current_engine_type = engine_type

        self.logger.info(
            f"Engine started successfully for model '{model_name}' "
            f"(PID: {engine_process.get_pid})"
        )

    async def _wait_for_engine_ready(
        self, engine_client: EngineClient, timeout: float
    ) -> bool:
        """
        Wait for engine to become ready by polling health check.

        Args:
            engine_client: EngineClient to check health
            timeout: Maximum time to wait in seconds

        Returns:
            True if engine becomes ready

        Raises:
            TimeoutError: If engine doesn't become ready within timeout
        """
        start_time = asyncio.get_event_loop().time()
        check_interval = 0.25  # Check every 0.25 seconds

        self.logger.info(f"Waiting for engine to become ready (timeout: {timeout}s)")

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed >= timeout:
                raise TimeoutError(
                    f"Engine did not become ready within {timeout} seconds"
                )

            # Check engine health
            try:
                if await engine_client.check_health():
                    self.logger.info(f"Engine became ready after {elapsed:.3f} seconds")
                    return True
            except Exception as e:
                self.logger.debug(f"Health check error (will retry): {e}")

            # Log progress
            self.logger.debug(f"Waiting for engine to be ready... {elapsed:.3f} sec")

            # Wait before next check
            await asyncio.sleep(check_interval)

    async def shutdown(self) -> None:
        """
        Shutdown the engine manager and stop any running engines.
        """
        self.logger.info("Shutting down EngineManager")

        async with self._lock:
            await self._stop_current_engine()

        self.logger.info("EngineManager shutdown complete")
