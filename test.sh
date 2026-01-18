#!/bin/bash
set -e

script_dir="$( cd "$( dirname "$0" )" && pwd )"
venv_dir="$script_dir/venv"

if [[ ! -d "$venv_dir" ]]; then
  echo "virtual env missing, run init.sh first!"
  exit 1
fi

. "$venv_dir/bin/activate"

# Check if any arguments were passed
if [[ $# -eq 0 ]]; then
  # No arguments - run all tests with verbose output
  "$venv_dir/bin/python" -m unittest discover -s tests -v
else
  # Pass arguments to unittest
  "$venv_dir/bin/python" -m unittest "$@"
fi
