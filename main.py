import argparse
import sys
from config.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM AWQ models with Huggingface Transformers and serve via Ollama/OpenAI compatible API.",
        add_help=False
    )
    parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the YAML config file"
    )

    args = parser.parse_args()

    try:
        config = Config(args.config)
        print("Config loaded successfully.")
        # TODO: Add further logic for model loading and API serving
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
