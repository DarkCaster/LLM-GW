#!/bin/bash
set -e
script_dir="$( cd "$( dirname "$0" )" && pwd )"

if [[ ! -d "$script_dir/venv" ]]; then
  "$script_dir/init.sh"
fi

"$script_dir/venv/bin/python" "$script_dir/download_hf_repo.py" "$@"
