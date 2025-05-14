import argparse
import os
from huggingface_hub import snapshot_download


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Download Model Repository")

    # Parse command line arguments
    parser.add_argument(
        "-r",
        "--repo",
        required=True,
        help="Specify HF repository name, example: runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Specify output path (full or relative), last component will be created",
    )

    args = parser.parse_args()

    # Implement the main program logic here
    print(f"Repo: {args.repo}")
    print(f"Output Path: {args.output}")

    # create output folder
    os.makedirs(args.output, exist_ok=True)

    # download to it with following method
    snapshot_download(repo_id=args.repo, local_dir=args.output)


if __name__ == "__main__":
    main()
