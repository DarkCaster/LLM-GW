import argparse
import os
import sys
import shutil
import logging
from pathlib import Path

# imports from my subpackages
import config
from utils.logger import setup_logging, get_logger


def main():
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

    try:
        logger.info("Starting LLM gateway")
        # start server here and wait for interrupt
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        logger.info("Stopping LLM gateway")
        # add any server termination tasks here
        temp_cleanup()


if __name__ == "__main__":
    main()
