import argparse
import os
import sys
import shutil
import logging
import signal
import asyncio
from typing import Optional

# imports from my subpackages
import config
from utils.logger import setup_logging, get_logger
from models.model_selector import ModelSelector
from engine.engine_manager import EngineManager


class LLMGateway:
    """Main LLM Gateway application."""

    def __init__(self, config_path: str):
        """
        Initialize LLM Gateway.

        Args:
            config_path: Path to Lua configuration file
        """
        self.config_path = config_path
        self.logger = get_logger(__name__)

        # Core components
        self.config_loader: Optional[config.ConfigLoader] = None
        self.cfg = None
        self.temp_dir: Optional[str] = None
        self.model_selector: Optional[ModelSelector] = None
        self.engine_manager: Optional[EngineManager] = None

        # Server components (to be implemented in Phase 3)
        self.server = None

        # Event loop and shutdown flag
        self.loop = asyncio.get_event_loop()
        self.shutdown_requested = False

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self.logger.info(
                f"Received signal {signum}, initiating graceful shutdown..."
            )
            self.shutdown_requested = True

            # Schedule shutdown in the event loop
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.shutdown())
                )
            else:
                # If loop isn't running yet, run shutdown directly
                self.loop.run_until_complete(self.shutdown())

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def initialize(self):
        """Initialize all components of the LLM Gateway."""
        try:
            # Setup logging with configured level if available
            log_level = self.cfg.get("server.log_level", "INFO").upper()
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            from utils.logger import setup_logging as setup_logging_func

            setup_logging_func(level=level_map.get(log_level, logging.INFO))

            self.logger.info(
                f"Initializing LLM Gateway with config: {self.config_path}"
            )

            # Load configuration
            self.config_loader = config.ConfigLoader(self.config_path)
            self.cfg = self.config_loader.cfg
            self.temp_dir = self.config_loader.temp_dir
            self.logger.info(f"Using temp directory: {self.temp_dir}")

            # Initialize GatewayServer (includes ModelSelector, EngineManager, and RequestHandler)
            from server.gateway_server import GatewayServer
            self.server = GatewayServer(self.cfg)
            self.logger.info("GatewayServer initialized")

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM Gateway: {e}")
            return False

    async def run(self):
        """Run the main application loop."""
        if not await self.initialize():
            self.logger.error("Initialization failed, cannot run")
            return 1

        self.logger.info("LLM Gateway is ready")

        try:
            # Start the GatewayServer and run until shutdown signal received
            await self.server.run()

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            return 1

        return 0

    async def shutdown(self):
        """Gracefully shutdown all components."""
        self.logger.info("Shutting down LLM Gateway...")

        # Stop GatewayServer (includes EngineManager shutdown)
        if self.server:
            try:
                await self.server.stop()
                self.logger.info("GatewayServer shutdown complete")
            except Exception as e:
                self.logger.error(f"Error shutting down GatewayServer: {e}")

        # Cleanup temp directory
        if self.temp_dir:
            try:
                self.logger.info(f"Cleaning up temp directory: {self.temp_dir}")
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp directory: {e}")
        self.logger.info("LLM Gateway shutdown complete")

    def temp_cleanup(self):
        """Cleanup temp directory (fallback for synchronous cleanup)."""
        try:
            self.logger.info(f"Cleaning up temp dir {self.temp_dir}")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp directory: {e}")


def main():
    """Main entry point for LLM Gateway."""
    parser = argparse.ArgumentParser(
        description="Run LLM-Gateway, manage LLM engines on demand per request"
    )
    parser.add_argument("-c", "--config", required=True, help="Lua configuration file")
    args = parser.parse_args()

    # Setup initial logging
    setup_logging()
    logger = get_logger(__name__)

    # Verify config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    # Create and run the gateway
    gateway = LLMGateway(args.config)

    # Setup signal handlers
    gateway.setup_signal_handlers()

    try:
        # Run the async main loop
        exit_code = asyncio.run(gateway.run())
        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Ensure cleanup on error
        gateway.temp_cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
