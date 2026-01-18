"""
GatewayServer - Main HTTP server for LLM gateway.

This module provides the main HTTP server implementation that listens on
configured addresses and routes requests to appropriate handlers.
"""

import asyncio
import re
from typing import Tuple, List, Optional

from aiohttp import web
import python_lua_helper

from models.model_selector import ModelSelector
from engine.engine_manager import EngineManager
from server.request_handler import RequestHandler
from utils.logger import get_logger


class GatewayServer:
    """Main HTTP server that listens on configured addresses and routes requests."""

    def __init__(self, cfg: python_lua_helper.PyLuaHelper):
        """
        Initialize GatewayServer with configuration.

        Args:
            cfg: PyLuaHelper configuration object
        """
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)

        # Parse server configuration
        self.listen_v4 = self.cfg.get("server.listen_v4", "127.0.0.1:7777")
        self.listen_v6 = self.cfg.get("server.listen_v6", "none")

        # Core components
        self.model_selector: Optional[ModelSelector] = None
        self.engine_manager: Optional[EngineManager] = None
        self.request_handler: Optional[RequestHandler] = None
        self.app: Optional[web.Application] = None

        # Server state
        self.runners: List[web.AppRunner] = []
        self.sites: List[web.TCPSite] = []
        self.is_running = False

        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all core components."""
        self.logger.info("Initializing GatewayServer components...")

        # Initialize ModelSelector
        self.model_selector = ModelSelector(self.cfg)
        self.logger.info(
            f"ModelSelector initialized with {len(self.model_selector.list_models())} models"
        )

        # Initialize EngineManager (singleton)
        self.engine_manager = EngineManager()
        self.logger.info("EngineManager initialized")

        # Initialize RequestHandler
        self.request_handler = RequestHandler(
            model_selector=self.model_selector,
            engine_manager=self.engine_manager,
            cfg=self.cfg,
        )
        self.logger.info("RequestHandler initialized")

    def _parse_listen_address(self, address: str) -> Tuple[str, int]:
        """
        Parse a listen address string into host and port.

        Args:
            address: Address string in format "host:port"

        Returns:
            Tuple of (host, port)

        Raises:
            ValueError: If address format is invalid
        """
        if address.lower() == "none":
            raise ValueError("Address is 'none', skipping")

        # Match IPv4/IPv6 address with port
        pattern = r"^(\[[0-9a-fA-F:]+\]|[^:]+):(\d+)$"
        match = re.match(pattern, address)

        if not match:
            raise ValueError(f"Invalid address format: {address}")

        host = match.group(1)
        port = int(match.group(2))

        # Remove brackets from IPv6 addresses
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]

        # Validate port range
        if not (1 <= port <= 65535):
            raise ValueError(f"Port {port} out of valid range (1-65535)")

        return host, port

    async def start(self) -> None:
        """
        Start the HTTP server.

        Creates web application, registers routes, and starts listening
        on configured addresses.

        Raises:
            RuntimeError: If server is already running or fails to start
        """
        if self.is_running:
            raise RuntimeError("Server is already running")

        self.logger.info("Starting GatewayServer...")

        try:
            # Create aiohttp web application
            self.app = web.Application()
            self._register_routes()

            # Create app runner
            runner = web.AppRunner(self.app)
            await runner.setup()
            self.runners.append(runner)

            # Start sites for configured addresses
            await self._start_listen_sites(runner)

            self.is_running = True
            self.logger.info("GatewayServer started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start GatewayServer: {e}")
            await self.stop()
            raise RuntimeError(f"Failed to start server: {e}")

    def _register_routes(self) -> None:
        """Register all API routes."""
        if not self.app or not self.request_handler:
            raise RuntimeError("Components not initialized")

        self.app.router.add_post(
            "/v1/chat/completions", self.request_handler.handle_chat_completion
        )
        self.app.router.add_post(
            "/v1/completions", self.request_handler.handle_completion
        )
        self.app.router.add_get("/v1/models", self.request_handler.handle_models_list)
        self.app.router.add_get(
            "/v1/models/{model_id}", self.request_handler.handle_model_info
        )

        # Add health check endpoint
        self.app.router.add_get("/health", self._handle_health_check)

        self.logger.debug("Routes registered successfully")

    async def _start_listen_sites(self, runner: web.AppRunner) -> None:
        """
        Start TCP sites for listening addresses.

        Args:
            runner: AppRunner instance

        Raises:
            RuntimeError: If no valid listen addresses are configured
        """
        sites_configured = False

        # Start IPv4 listener
        if self.listen_v4.lower() != "none":
            try:
                host, port = self._parse_listen_address(self.listen_v4)
                site = web.TCPSite(runner, host, port)
                await site.start()
                self.sites.append(site)
                sites_configured = True
                self.logger.info(f"Listening on IPv4: {host}:{port}")
            except ValueError as e:
                self.logger.error(f"Invalid IPv4 address '{self.listen_v4}': {e}")
            except Exception as e:
                self.logger.error(f"Failed to start IPv4 listener: {e}")

        # Start IPv6 listener
        if self.listen_v6.lower() != "none":
            try:
                host, port = self._parse_listen_address(self.listen_v6)
                site = web.TCPSite(runner, host, port)
                await site.start()
                self.sites.append(site)
                sites_configured = True
                self.logger.info(f"Listening on IPv6: {host}:{port}")
            except ValueError as e:
                self.logger.error(f"Invalid IPv6 address '{self.listen_v6}': {e}")
            except Exception as e:
                self.logger.error(f"Failed to start IPv6 listener: {e}")

        if not sites_configured:
            raise RuntimeError(
                "No valid listen addresses configured. "
                "Check server.listen_v4 and server.listen_v6 in configuration."
            )

    async def stop(self) -> None:
        """Stop the HTTP server gracefully."""
        if not self.is_running:
            self.logger.debug("Server is not running, nothing to stop")
            return

        self.logger.info("Stopping GatewayServer...")

        # Stop all sites
        for site in self.sites:
            try:
                await site.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping site: {e}")

        # Stop all runners
        for runner in self.runners:
            try:
                await runner.cleanup()
            except Exception as e:
                self.logger.warning(f"Error cleaning up runner: {e}")

        # Shutdown engine manager
        if self.engine_manager:
            try:
                await self.engine_manager.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down EngineManager: {e}")

        # Clear state
        self.sites.clear()
        self.runners.clear()
        self.is_running = False

        self.logger.info("GatewayServer stopped successfully")

    async def run(self) -> None:
        """
        Run the server until interrupted.

        This method starts the server and then waits forever (or until
        a shutdown signal is received).
        """
        try:
            await self.start()

            # Log available models
            if self.model_selector:
                models = self.model_selector.list_models()
                self.logger.info(f"Available models: {', '.join(models)}")

            # Keep running until interrupted
            while self.is_running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Server run error: {e}")
            raise
        finally:
            await self.stop()

    async def _handle_health_check(self, request: web.Request) -> web.Response:
        """
        Handle health check endpoint.

        Args:
            request: HTTP request

        Returns:
            Health check response
        """
        health_status = {
            "status": "ok" if self.is_running else "down",
            "service": "llm-gateway",
            "version": "1.0.0",
            "models_available": len(self.model_selector.list_models())
            if self.model_selector
            else 0,
            "engine_running": self.engine_manager.get_current_state()["engine_running"]
            if self.engine_manager
            else False,
        }

        status_code = 200 if self.is_running else 503
        return web.json_response(health_status, status=status_code)

    def get_server_info(self) -> dict:
        """
        Get server information for monitoring/debugging.

        Returns:
            Dictionary with server information
        """
        return {
            "running": self.is_running,
            "listen_v4": self.listen_v4,
            "listen_v6": self.listen_v6,
            "models_available": len(self.model_selector.list_models())
            if self.model_selector
            else 0,
            "engine_state": self.engine_manager.get_current_state()
            if self.engine_manager
            else {},
            "active_connections": len(self.runners) if self.runners else 0,
        }
