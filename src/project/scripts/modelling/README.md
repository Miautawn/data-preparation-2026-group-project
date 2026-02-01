# Model Training Scripts

Here you can use the scripts to train the models for next heart-rate prediction task

## How  to run

The following artifacts are needed to train the model:

- `train_preprocessed.parquet` - prprocessed training dataset
- `val_preprocessed.parquet` - perprocessed validation dataset

You can generate these artifacts by running [4_dataset_preprocessing.py](../dataset/4_dataset_preprocessing.py) script or supplying your own.

> [!NOTE]
> If you wish to use custom training and validation datasets, make sure to edit the global variables `SOURCE_TRAIN_PATH` and `SOURCE_VAL_PATH` inside the `training.py` script

Finally, you can run the script via:
```bash
uv run python training.py
```

**Output**:
- `.cache/fitrec_model.pt` - a fully saved pytorch model, which you can easily load using:
```python
import torch

model = torch.load("fitrec_model.pt", only_weights=False)
```
