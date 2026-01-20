#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")

"$SCRIPT_DIR/dataset_download.sh"

uv run python "$SCRIPT_DIR/dataset_preprocessing.py"
