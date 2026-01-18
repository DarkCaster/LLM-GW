# server/gateway_server.py

import asyncio
from aiohttp import web
from typing import Optional, List, Tuple
from utils.logger import get_logger
from models import ModelSelector
from engine import EngineManager
from .request_handler import RequestHandler


class GatewayServer:
    """
    Main HTTP server that listens and routes requests.
    """

    def __init__(self, cfg):
        """
        Initialize the gateway server.

        Args:
            cfg: PyLuaHelper configuration object
        """
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)

        # Parse server configuration
        self.listen_v4 = self.cfg.get("server.listen_v4", "none")
        self.listen_v6 = self.cfg.get("server.listen_v6", "none")

        # Initialize components
        self.model_selector = ModelSelector(cfg)
        self.engine_manager = EngineManager()
        self.request_handler = RequestHandler(
            self.model_selector, self.engine_manager, cfg
        )

        # Server state
        self.app: Optional[web.Application] = None
        self.runners: List[web.AppRunner] = []
        self.sites: List[web.TCPSite] = []

    async def start(self) -> None:
        """
        Start the HTTP server and begin listening.
        """
        self.logger.info("Starting Gateway Server")

        # Create aiohttp application
        self.app = web.Application()

        # Register routes
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

        self.logger.info("Routes registered")

        # Create runners for IPv4 and IPv6
        runner = web.AppRunner(self.app)
        await runner.setup()
        self.runners.append(runner)

        # Start IPv4 listener if configured
        if self.listen_v4 and self.listen_v4.lower() != "none":
            try:
                host, port = self._parse_listen_address(self.listen_v4)
                site = web.TCPSite(
                    runner, host, port, reuse_address=True, reuse_port=True
                )
                await site.start()
                self.sites.append(site)
                self.logger.info(f"Listening on IPv4: {host}:{port}")
            except Exception as e:
                self.logger.error(
                    f"Failed to start IPv4 listener on {self.listen_v4}: {e}"
                )
                raise

        # Start IPv6 listener if configured
        if self.listen_v6 and self.listen_v6.lower() != "none":
            try:
                host, port = self._parse_listen_address(self.listen_v6)
                site = web.TCPSite(
                    runner, host, port, reuse_address=True, reuse_port=True
                )
                await site.start()
                self.sites.append(site)
                self.logger.info(f"Listening on IPv6: {host}:{port}")
            except Exception as e:
                self.logger.error(
                    f"Failed to start IPv6 listener on {self.listen_v6}: {e}"
                )
                # Don't raise here, IPv6 might not be available

        if not self.sites:
            raise RuntimeError("No listeners were started. Check configuration.")

        self.logger.info(f"Gateway Server started with {len(self.sites)} listener(s)")

    async def stop(self) -> None:
        """
        Stop the HTTP server and clean up resources.
        """
        self.logger.info("Stopping Gateway Server")

        # Stop all sites
        for site in self.sites:
            await site.stop()

        self.sites.clear()

        # Clean up runners
        for runner in self.runners:
            await runner.cleanup()

        self.runners.clear()

        # Shutdown engine manager
        await self.engine_manager.shutdown()

        # Clean up application
        if self.app:
            await self.app.shutdown()
            await self.app.cleanup()

        self.logger.info("Gateway Server stopped")

    async def run(self) -> None:
        """
        Start the server and wait forever until interrupted.
        """
        # Start the server
        await self.start()

        try:
            # Wait forever
            self.logger.info("Server is running. Press Ctrl+C to stop.")
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.logger.info("Server run cancelled")
        finally:
            # Stop on interrupt
            await self.stop()

    def _parse_listen_address(self, address: str) -> Tuple[str, int]:
        """
        Parse "host:port" string into components.

        Args:
            address: Address string in "host:port" format

        Returns:
            Tuple of (host, port)

        Raises:
            ValueError: If address format is invalid
        """
        if not address or address.lower() == "none":
            raise ValueError("Invalid address: empty or 'none'")

        parts = address.rsplit(":", 1)

        if len(parts) != 2:
            raise ValueError(f"Invalid address format: {address}. Expected 'host:port'")

        host = parts[0].strip()
        port_str = parts[1].strip()

        # Handle IPv6 addresses in brackets
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]

        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port number: {port_str}")

        if port < 1 or port > 65535:
            raise ValueError(f"Port number out of range: {port}")

        return (host, port)
