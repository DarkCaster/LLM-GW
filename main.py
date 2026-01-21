import argparse
import sys
import shutil
import asyncio
import aiohttp
import python_lua_helper

# imports from my subpackages
import config
from utils.logger import setup_logging, get_logger
from engine import EngineManager
from models import ModelSelector
from server import RequestHandler, GatewayServer


async def async_main(cfg: python_lua_helper.PyLuaHelper):
    """
    Async main function that sets up and runs the server.

    Args:
        cfg: PyLuaHelper configuration object
    """
    logger = get_logger("Main(async)")

    # Create aiohttp ClientSession for HTTP communication
    async with aiohttp.ClientSession() as session:
        # Initialize components
        engine_manager = EngineManager(session, cfg)
        model_selector = ModelSelector(engine_manager, cfg)
        request_handler = RequestHandler(model_selector, engine_manager, cfg)
        gateway_server = GatewayServer(request_handler, cfg)

        try:
            # Run the server (blocks until interrupted)
            await gateway_server.run()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
        finally:
            # TODO: shutdown idle watchdog
            # Shutdown engine manager
            await engine_manager.shutdown()


def main():
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
