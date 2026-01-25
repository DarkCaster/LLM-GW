import argparse
import sys
import shutil
import asyncio
import aiohttp
import python_lua_helper

# imports from my subpackages
import config
import engine
import models
import server
from utils.logger import setup_logging, get_logger


async def async_main(cfg: python_lua_helper.PyLuaHelper) -> None:
    """
    Async main function that sets up and runs the server.

    Args:
        cfg: PyLuaHelper configuration object
    """
    logger = get_logger("Main(async)")

    # Create aiohttp ClientSession for HTTP communication
    async with aiohttp.ClientSession() as session:
        # Initialize components
        engine_manager = engine.EngineManager(session, cfg)
        model_selector = models.ModelSelector(engine_manager, cfg)
        request_handler = server.RequestHandler(
            model_selector, engine_manager, server.IdleWatchdog(), cfg
        )
        gateway_server = server.GatewayServer(request_handler, cfg)

        try:
            # Start the server
            await gateway_server.start()
            # Wait forever (until interrupted)
            logger.info("Press Ctrl+C to stop.")
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            logger.info("Received cancellation signal")
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
        finally:
            # Stop serving new requests
            request_handler.stop()
            # Shutdown engine manager
            await engine_manager.shutdown()
            await request_handler.shutdown()
            await gateway_server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-Gateway, manage LLM engines on demand per request"
    )
    parser.add_argument("-c", "--config", required=True, help="Lua configuration file")
    args = parser.parse_args()

    # Setup logging initially with default level
    setup_logging()
    logger = get_logger("Main")

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

    try:
        logger.info("Starting LLM gateway")
        # Run the async main function
        asyncio.run(async_main(cfg))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Stopping LLM gateway")
        temp_cleanup()


if __name__ == "__main__":
    main()
