## Baked Artifacts
This directory is meant for downloading and storing baked artifacts (i.e. prepared datasets/trained models) such that the reader could easily run error injection, cleaning and analysis scripts!

> [!NOTE]
> If you wish to recreate these artifacts from scratch, follow the instructions in [scripts/dataset](../scripts/dataset/README.md) and [scripts/modelling](../scripts/modelling/README.md).

## How to run

Simply run this python script, which will download and extract necessary artifacts for all data segments:
```bash
uv run python download_baked_artifacts.py
```

After running it, this directory will be populated with 3 data segment directories (i.e. `walking`/`running`/`biking`) each with their own set of artifacts:

- `test_raw.parquet` - a simple raw 20% split of the dataset (which you can experiment injecting data on)
- `user_standard_scaler.pkl` - pickled python object used for standardizing the sequential features, fitted on train split.
- `static_ordinal_encoder.pkl` - pickled python object used for ordinally encoding static features (e.g. user_id, sport, gender), fitted on train split.
- `fitrec_model.pt` - model trained on all sequential features + previous heart-rates in auto regressive manner. Achieves great performance. Pass use_heartrate_input = True when creating FitRecDataset dataset.

After you have the necessary artifacts extracted, please follow the rest of instructions in the root [README](../../../README.md) for reproducing the rest of the project steps.

> [!NOTE]
> You can test the model out directly by following instructions in [this notebook](../notebooks/test_inference.ipynb)
