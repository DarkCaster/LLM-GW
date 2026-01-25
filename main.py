import argparse
import sys
import shutil
import asyncio
import aiohttp
import python_lua_helper
import config
import engine
import models
import server
import logger


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
        temp_dir = config_loader.temp_dir
        log.info(f"Using temp directory: {temp_dir}")
    except Exception as e:
        log.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    def temp_cleanup():
        if cfg.get_bool("profile.temp_cleanup", True):
            log.info(f"Cleaning up temp dir {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

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
        temp_cleanup()


if __name__ == "__main__":
    main()
