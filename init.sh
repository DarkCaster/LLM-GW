#!/bin/bash
set -e
script_dir="$( cd "$( dirname "$0" )" && pwd )"

if [[ ! -d "$script_dir/venv" ]]; then
  virtualenv "$script_dir/venv"
fi

"$script_dir/venv/bin/pip" --require-virtualenv install --upgrade -r "$script_dir/requirements.txt"
