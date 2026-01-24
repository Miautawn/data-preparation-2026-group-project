#!/bin/sh

# get current script location
SCRIPT_DIR=$(dirname "$(realpath "$0")")

mkdir -p "$SCRIPT_DIR/.cache/"

# take the processed_endomondoHR_proper_interpolate.npy from here https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html
curl https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/processed_endomondoHR_proper_interpolate.npy \
    --output "$SCRIPT_DIR/.cache/processed_endomondoHR_proper_interpolate.npy" \
    --progress-bar

# take the endomondoHR_proper.json from here https://www.kaggle.com/datasets/pypiahmad/endomondo-fitness-trajectories
uv run kaggle datasets download \
    -d pypiahmad/endomondo-fitness-trajectories \
    --file endomondoHR_proper.json \
    -p "$SCRIPT_DIR/.cache/"

unzip "$SCRIPT_DIR/.cache/endomondoHR_proper.json.zip" -d "$SCRIPT_DIR/.cache/" >> /dev/null 2>&1
rm "$SCRIPT_DIR/.cache/endomondoHR_proper.json.zip"
