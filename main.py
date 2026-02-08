import argparse
import sys
import asyncio
import aiohttp
import python_lua_helper
import config
import engine
import models
import server
import logger
from server.dump_writer import clear_dumps_directory


async def async_main(cfg: python_lua_helper.PyLuaHelper) -> None:
    """
    Async main function that sets up and runs the server.

    Args:
        cfg: PyLuaHelper configuration object
    """
    log = logger.get_logger("Main(async)")

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
            log.info("Press Ctrl+C to stop.")
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            log.info("Received cancellation signal")
        except KeyboardInterrupt:
            log.info("Received keyboard interrupt")
        except Exception as e:
            log.error(f"Server error: {e}", exc_info=True)
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
    logger.setup_logging()
    log = logger.get_logger("Main")

    # Load configuration
    try:
        config_loader = config.ConfigLoader(args.config)
        cfg = config_loader.cfg
    except Exception as e:
        log.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Clear dump files on startup if configured
    dumps_dir = cfg.get("server.dumps_dir")
    clear_dumps = cfg.get_bool("server.clear_dumps_on_start", False)

    if dumps_dir and clear_dumps:
        log.info(f"Clearing dump files from directory: {dumps_dir}")
        clear_dumps_directory(dumps_dir)

    try:
        log.info("Starting LLM gateway")
        # Run the async main function
        asyncio.run(async_main(cfg))
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("Stopping LLM gateway")


if __name__ == "__main__":
    main()
