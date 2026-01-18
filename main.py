import argparse
import sys
import shutil
import asyncio
import signal

# imports from my subpackages
import config
from utils.logger import setup_logging, get_logger
from server import GatewayServer


# Global flag for graceful shutdown
shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger = get_logger(__name__)
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


async def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-Gateway, manage LLM engines on demand per request"
    )
    parser.add_argument("-c", "--config", required=True, help="Lua configuration file")
    args = parser.parse_args()

    # Setup logging initially with default level
    setup_logging()
    logger = get_logger(__name__)

    # Load configuration
    try:
        config_loader = config.ConfigLoader(args.config)
        cfg = config_loader.cfg
        temp_dir = config_loader.temp_dir
        logger.info(f"Using temp directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    def temp_cleanup():
        if cfg.get_bool("profile.temp_cleanup", True):
            logger.info(f"Cleaning up temp dir {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Create gateway server
    server = None

    try:
        logger.info("Starting LLM gateway")

        # Initialize server
        server = GatewayServer(cfg)

        # Start server (this will set up listeners)
        await server.start()

        # Wait for shutdown signal
        logger.info("Server is running. Press Ctrl+C to stop.")
        await shutdown_event.wait()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Stopping LLM gateway")
        # Stop server if it was created
        if server:
            try:
                await server.stop()
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
        temp_cleanup()


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Already handled in main()
        pass
