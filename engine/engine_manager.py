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


# NOTE: assumed that EngineManager's instance public methods run one at a time from external flow, so, we do not need to use asyncio.Lock
class EngineManager:
    """
    Coordinate engine lifecycle - stop old engines, start new ones, track state.
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
        self.logger = get_logger(self.__class__.__name__)
        self.session = session
        self.cfg = cfg
        self._is_disposed = False
        # Engine state
        self._current_engine_process: Optional[EngineProcess] = None
        self._current_engine_client: Optional[EngineClient] = None
        self._current_model_name: Optional[str] = None
        self._current_config: Optional[Dict[str, Any]] = None
        self._current_engine_type: Optional[str] = None
        self._current_idle_timeout: float = sys.float_info.max
        self.logger.info("EngineManager initialized")

    def _check_model_configuration(
        self, model_name: str, required_config: dict
    ) -> bool:
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

    def _get_model_index(self, model_name: str, raise_if_not_found: bool) -> int:
        model_index = None
        for i in self.cfg.get_table_seq("models"):
            if self.cfg.get(f"models.{i}.name") == model_name:
                model_index = i
                break
        if model_index is None:
            if raise_if_not_found:
                raise ValueError(f"Model '{model_name}' not found in configuration")
            else:
                self.logger.error(
                    f"Model '{self._current_model_name}' not found in configuration"
                )
                return -1
        return model_index

    def _check_llamacpp_config(self, required_config: dict) -> bool:
        """
        Check if current llamacpp engine configuration matches requirements.

        Args:
            required_config: Required configuration dictionary

        Returns:
            True if current engine is llamacpp type (stub implementation)
        """
        if not self._current_config or not required_config:
            self.logger.info("No engine running")
            return False
        # check cases when we requested config context_estimation
        if required_config.get("operation", "unknown") == "context_estimation":
            if self._current_config.get("operation", "unknown") == "context_estimation":
                # currently loaded model was already configured specifically for context_estimation
                self.logger.debug(
                    "Running engine was already configured for context estimation, reusing"
                )
                return True
            else:
                # check can we actually use currently loaded model configuration for context estimation
                # get variant index from current configuration (return false if no index)
                variant_index = self._current_config.get("variant_index")
                if variant_index is None:
                    self.logger.error("No variant index detected for running engine")
                    return False
                # get model index from cfg for current model name, return false if we cannot detect it
                model_index = self._get_model_index(self._current_model_name, False)
                if model_index < 0:
                    self.logger.error("Internal error (1)!")
                    return False
                # get `tokenize` value from config for model with known index and variant index, return false if tokenize is false
                if not self.cfg.get_bool(
                    f"models.{model_index}.variants.{variant_index}.tokenize", False
                ):
                    self.logger.info(
                        "Running engine do not support tokenization queries"
                    )
                    return False
                # currently loaded model is suitable
                self.logger.debug(
                    "Currently running engine is suitable for context estimation, reusing"
                )
                return True
        elif required_config.get("operation", "unknown") == "text_query":
            # we can use model loaded
            if (
                self._current_config.get("operation", "unknown") == "text_query"
                or self._current_config.get("operation", "unknown")
                == "context_estimation"
            ):
                # check can we actually use currently loaded model configuration for text query
                context_required = required_config.get(
                    "context_size_required", sys.maxsize
                )
                # get variant index from current configuration (return false if no index)
                variant_index = self._current_config.get("variant_index")
                if variant_index is None:
                    self.logger.error("No variant index detected for running engine")
                    return False
                # get model index from cfg for current model name, return false if we cannot detect it
                model_index = self._get_model_index(self._current_model_name, False)
                if model_index < 0:
                    self.logger.error("Internal error (2)!")
                    return False
                # get `context` value from cfg (with fallback 0), compare with context_required return false if context value is smaller
                current_context = self.cfg.get_int(
                    f"models.{model_index}.variants.{variant_index}.context", 0
                )
                if current_context < context_required:
                    self.logger.info(
                        "Currently running engine context size is not sufficient for text query"
                    )
                    return False
                # currently loaded model is suitable
                return True
        # For any other case - currently loaded model is not suitable
        return False

    async def ensure_engine(
        self, model_name: str, required_config: dict
    ) -> tuple[EngineClient, float]:
        """
        Ensure the correct engine is running with the required configuration.

        Args:
            model_name: Name of the model to load
            required_config: Configuration dictionary with variant_index

        Returns:
            EngineClient instance for the running engine with proposed idle timeout

        Raises:
            ValueError: If model not found, engine type not supported, or config invalid
            TimeoutError: If engine fails to become ready
        """
        if self._is_disposed:
            raise RuntimeError("EngineManager is shutdown")
        # Check if current engine configuration is suitable
        if self._check_model_configuration(model_name, required_config):
            # Verify health
            if self._current_engine_client is not None:
                if await self._current_engine_client.check_health():
                    self.logger.debug(
                        f"Current engine for model '{model_name}' is already running and healthy"
                    )
                    return self._current_engine_client, self._current_idle_timeout
                else:
                    self.logger.info(
                        f"Current engine for model '{model_name}' failed health check"
                    )
        # Find model in configuration
        model_index = self._get_model_index(model_name, True)
        # Get engine type for the model
        cfg_engine_type = self.cfg.get(f"models.{model_index}.engine")
        # NOTE: engine specifig setup here:
        if cfg_engine_type == "llama.cpp":
            # Iterate over model's variants and select first suitable variant
            context_required = required_config.get("context_size_required", sys.maxsize)
            variant_index = None
            for i in self.cfg.get_table_seq(f"models.{model_index}.variants"):
                variant_context = self.cfg.get_int(
                    f"models.{model_index}.variants.{i}.context", 0
                )
                if variant_context >= context_required:
                    variant_index = i
                    self.logger.info(
                        f"Selected variant {variant_index} with context size {variant_context}"
                    )
                    break
            if variant_index is None:
                raise ValueError(
                    f"No suitable variant found for model '{model_name}' "
                    f"with required context size {context_required}"
                )
            required_config["variant_index"] = variant_index
        else:
            raise ValueError(f"Engine type '{cfg_engine_type}' not supported.")
        # Stop and start selected engine
        self.logger.info(f"Starting new engine for model '{model_name}'")
        await self.stop_current_engine()
        await self._start_new_engine(model_name, required_config, cfg_engine_type)
        return self._current_engine_client, self._current_idle_timeout

    async def stop_current_engine(self) -> None:
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
        self._current_idle_timeout = sys.float_info.max
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
        # NOTE: engine specifig startup here:
        engine_idle_timeout = sys.float_info.max
        if engine_type == "llama.cpp":
            # Find model in configuration
            model_index = self._get_model_index(model_name, True)
            # Extract variant_index from required_config
            variant_index = required_config.get("variant_index")
            if variant_index is None:
                raise ValueError("variant_index not specified in required_config")
            # Extract binary, args, and connect URL
            variant_key = f"models.{model_index}.variants.{variant_index}"
            binary = self.cfg.get(f"{variant_key}.binary")
            if not binary:
                raise ValueError(f"Binary path not found for variant {variant_index}")
            connect_url = self.cfg.get(f"{variant_key}.connect")
            if not connect_url:
                raise ValueError(f"Connect URL not found for variant {variant_index}")
            args = self.cfg.get_list(f"{variant_key}.args")
            self.logger.info(
                f"Starting engine: binary={binary}, connect={connect_url}, "
                f"args count={len(args)}"
            )
            # Get timeouts
            engine_startup_timeout = self.cfg.get_float(
                f"{variant_key}.engine_startup_timeout"
            )
            health_check_timeout = self.cfg.get_float(
                f"{variant_key}.health_check_timeout"
            )
            engine_idle_timeout = self.cfg.get_float(
                f"{variant_key}.engine_idle_timeout"
            )
            # Create and start EngineProcess
            engine_client = LlamaCppEngine(
                self.session, connect_url, health_check_timeout
            )
            engine_process = EngineProcess(binary, args)
            await engine_process.start()
            # Wait for engine to become ready
            try:
                await self._wait_for_engine_ready(
                    engine_client, timeout=engine_startup_timeout
                )
            except Exception as e:
                # If engine fails to become ready, stop the process
                self.logger.error(f"Engine failed to become ready: {e}")
                await engine_process.stop()
                raise
        else:
            raise ValueError(f"Engine type '{engine_type}' not supported.")
        # Store engine state
        self._current_engine_process = engine_process
        self._current_engine_client = engine_client
        self._current_model_name = model_name
        self._current_config = required_config
        self._current_engine_type = engine_type
        self._current_idle_timeout = engine_idle_timeout
        self.logger.info(
            f"Engine started successfully for model '{model_name}', (PID: {engine_process.get_pid})"
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
        if self._is_disposed:
            self.logger.debug("EngineManager already shutdown")
            return
        self._is_disposed = True
        await self.stop_current_engine()
        self.logger.info("EngineManager shutdown complete")
