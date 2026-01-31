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

## How to run

To fully reproduce the project workflow, you need to re-run all the steps:
> [!WARNING]
> First steps can be time consuming, read the instructions below first!
* [dataset preparation](./src/project/scripts/dataset/README.md)
* [model training](./src/project/scripts/modelling/README.md)
* [error injection](./src/project/corruption/)
* [dataset cleaning](./src/project/cleaning/)
* analysis

However, `dataset preparation` and `model training` can take up some time, thus we offer checkpointed artifacts which you can easily downalod by following the instructions [here](./src/project/baked_artifacts/README.md). This will allow you to more effortlessly run the rest of the steps.

After you have downloaded (or re-generated) the dataset and ML model artifacts, you can perform error injection by running the command below. The details of the generated dataset in [this file](./src/project/corruption/README.md) :
```bash
python src/project/corruption/main_error_injection.py
```

The, you can perform data cleaning operations via:
**ADD INSTRUCTIONS HERE**

And finally, you can re-run the model inference and analysis via:
**ADD INSTRUCTIONS HERE**
