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