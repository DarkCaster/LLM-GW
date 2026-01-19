# server/gateway_server.py

import logging
import asyncio
import aiohttp.web
from typing import List, Optional, Tuple
import python_lua_helper
from .request_handler import RequestHandler


class GatewayServer:
    """
    Main HTTP server that listens on configured addresses and routes requests.
    """

    def __init__(
        self, request_handler: RequestHandler, cfg: python_lua_helper.PyLuaHelper
    ):
        """
        Initialize GatewayServer.

        Args:
            request_handler: RequestHandler instance for processing requests
            cfg: PyLuaHelper configuration object
        """
        self.request_handler = request_handler
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

        # Server state
        self.app: Optional[aiohttp.web.Application] = None
        self.runners: List[aiohttp.web.AppRunner] = []
        self.sites: List[aiohttp.web.TCPSite] = []

        self.logger.info("GatewayServer initialized")

    async def start(self) -> None:
        """
        Start the HTTP server on configured addresses.

        Raises:
            ValueError: If no valid listen addresses are configured
        """
        self.logger.info("Starting GatewayServer")

        # Create aiohttp application
        self.app = aiohttp.web.Application()

        # Register routes
        self.app.router.add_get("/v1/models", self.request_handler.handle_models_list)
        self.app.router.add_post(
            "/v1/chat/completions", self.request_handler.handle_request
        )

        self.logger.info("Routes registered")

        # Parse listen addresses from config
        listen_addresses = []

        # Check IPv4 address
        listen_v4 = self.cfg.get("server.listen_v4", "none")
        if listen_v4 != "none":
            try:
                host, port = self._parse_address(listen_v4)
                listen_addresses.append(("ipv4", host, port))
                self.logger.info(f"Will listen on IPv4: {host}:{port}")
            except Exception as e:
                self.logger.error(f"Failed to parse IPv4 address '{listen_v4}': {e}")
                raise ValueError(f"Invalid IPv4 listen address: {listen_v4}") from e

        # Check IPv6 address
        listen_v6 = self.cfg.get("server.listen_v6", "none")
        if listen_v6 != "none":
            try:
                host, port = self._parse_address(listen_v6)
                listen_addresses.append(("ipv6", host, port))
                self.logger.info(f"Will listen on IPv6: {host}:{port}")
            except Exception as e:
                self.logger.error(f"Failed to parse IPv6 address '{listen_v6}': {e}")
                raise ValueError(f"Invalid IPv6 listen address: {listen_v6}") from e

        # Ensure at least one address is configured
        if not listen_addresses:
            raise ValueError(
                "No valid listen addresses configured. "
                "Set server.listen_v4 or server.listen_v6 in configuration."
            )

        # Create and start runners for each address
        for addr_type, host, port in listen_addresses:
            try:
                runner = aiohttp.web.AppRunner(self.app)
                await runner.setup()
                self.runners.append(runner)

                site = aiohttp.web.TCPSite(runner, host, port)
                await site.start()
                self.sites.append(site)

                self.logger.info(f"Server listening on {addr_type} {host}:{port}")
            except Exception as e:
                self.logger.error(
                    f"Failed to start server on {addr_type} {host}:{port}: {e}"
                )
                # Cleanup any runners that were started
                await self.stop()
                raise

        self.logger.info(
            f"GatewayServer started successfully on {len(self.sites)} address(es)"
        )

    async def stop(self) -> None:
        """
        Stop the HTTP server and cleanup resources.
        """
        self.logger.info("Stopping GatewayServer")

        # Stop all sites
        for site in self.sites:
            try:
                await site.stop()
            except Exception as e:
                self.logger.error(f"Error stopping site: {e}")

        # Cleanup all runners
        for runner in self.runners:
            try:
                await runner.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up runner: {e}")

        # Clear state
        self.sites.clear()
        self.runners.clear()
        self.app = None

        self.logger.info("GatewayServer stopped")

    async def run(self) -> None:
        """
        Start the server and run until interrupted.

        Handles graceful shutdown on interrupt.
        """
        try:
            # Start the server
            await self.start()

            # Wait forever (until interrupted)
            self.logger.info("Server is running. Press Ctrl+C to stop.")
            await asyncio.Event().wait()

        except asyncio.CancelledError:
            self.logger.info("Server received cancellation signal")
        except KeyboardInterrupt:
            self.logger.info("Server received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Unexpected error in server run loop: {e}")
            raise
        finally:
            # Graceful shutdown
            await self.stop()

    def _parse_address(self, address: str) -> Tuple[str, int]:
        """
        Parse address string in format "host:port".

        Args:
            address: Address string like "127.0.0.1:7777" or "[::1]:8080"

        Returns:
            Tuple of (host, port)

        Raises:
            ValueError: If address format is invalid
        """
        # Handle IPv6 addresses with brackets like [::1]:8080
        if address.startswith("["):
            # IPv6 address
            bracket_end = address.find("]")
            if bracket_end == -1:
                raise ValueError(f"Invalid IPv6 address format: {address}")

            host = address[1:bracket_end]
            port_part = address[bracket_end + 1 :]

            if not port_part.startswith(":"):
                raise ValueError(f"Invalid IPv6 address format: {address}")

            try:
                port = int(port_part[1:])
            except ValueError:
                raise ValueError(f"Invalid port in address: {address}")

        else:
            # IPv4 address or hostname
            parts = address.rsplit(":", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid address format: {address}")

            host = parts[0]
            try:
                port = int(parts[1])
            except ValueError:
                raise ValueError(f"Invalid port in address: {address}")

        # Validate port range
        if port < 1 or port > 65535:
            raise ValueError(f"Port out of range (1-65535): {port}")

        return host, port
