# data-preparation-2026-group-project
Group project for UVA "Data Preparation" course 2026 (Group 10)

## Setup
This project uses [uv](https://github.com/astral-sh/uv) for python project dependency and environment managment.

To download the dependencies and instantiate the project virtual environment, simply run:
```bash
uv sync
```

> [!NOTE]
> UV uses `pyproject.toml` and `uv.lock` to track and lock the dependencise, so make sure to commit them!

Then, setup the pre-commit hooks by running:
```
pre-commit install
```

## Project structure
We use standard python "package/library" project structure:
```
src
└── project
    ├── main.py
    ├── notebooks
    │   └── main.ipynb
    └── utils
        ├── __init__.py
        └── my_utils.py
```


### for running data corruption script:
uv run python src/project/corruption/array_corruption/corrupt_data.py \
  --input  src/project/temp/biking/biking_test_raw.parquet \
  --output src/project/temp/biking/biking_test_raw_corrupted.parquet \
  --method random \
  --std-scale 3.0 \
  --row-fraction 1.0 \
  --segment-fraction 0.2 \
  --num-segments 2 \
  --seed 42 \
  --columns heart_rate derived_speed altitude

## batch clean all files
uv run python src/project/cleaning/cleaning_generic.py \
  --root src/project/temp \
  --pattern "**/*_test_raw_corrupted.parquet"

## advanced clean a file
uv run python src/project/cleaning/advanced_cleaning.py \
  --input  src/project/temp/biking/biking_test_raw_corrupted.parquet \
  --fit-on src/project/temp/biking/biking_test_raw.parquet \
  --output src/project/temp/biking/biking_test_raw_corrupted_cleaned.parquet \
  --speed-max 60 \
  --speed-max-jump 15 \
  --max-gap 60 \
  --ema-alpha 0.12


