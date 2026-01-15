import argparse
import os
import sys
import shutil
from pathlib import Path

# imports from my subpackages
import config


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-Gateway, manage LLM engines on demand per request"
    )
    parser.add_argument("-c", "--config", required=True, help="Lua configuration file")
    args = parser.parse_args()

    # Load configuration
    try:
        config_loader = config.ConfigLoader(args.config)
        cfg = config_loader.cfg
        temp_dir = config_loader.temp_dir
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)

    print(f"Using temp directory: {temp_dir}")
    # os.makedirs(temp_dir, exist_ok=True)

    # def temp_cleanup():
    #    if cfg.get_bool("profile.temp_cleanup", True):
    #        print(f"[{args.job_id}]: cleaning up temp dir")
    #        shutil.rmtree(temp_dir, ignore_errors=True)

    try:
        # Cleanup temporary directory
        # temp_cleanup()
        print("Shuting down server")

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
