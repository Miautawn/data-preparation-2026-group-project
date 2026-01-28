#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")


echo "1. Downloading the dataset artifacts..."
"$SCRIPT_DIR/1_dataset_download.sh"

echo "2. Assembling the dataset..."
uv run python "$SCRIPT_DIR/2_dataset_preparation.py"

echo "3. Splitting the dataset into train/val/test..."
uv run python "$SCRIPT_DIR/3_dataset_splitting.py"

echo "4. Standardizing and encoding the features..."
uv run python "$SCRIPT_DIR/4_dataset_preprocessing.py"
