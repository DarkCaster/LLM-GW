#!/bin/bash
set -e

script_dir="$( cd "$( dirname "$0" )" && pwd )"
venv_dir="$script_dir/venv"

if [[ ! -d "$venv_dir" ]]; then
  echo "virtual env missing, run init.sh first!"
  exit 1
fi

. "$venv_dir/bin/activate"

"$venv_dir/bin/python" "$script_dir/main.py" "$@"
