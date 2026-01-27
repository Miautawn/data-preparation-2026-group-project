# Dataset generation scripts
These scripts allow you to download the source data and preprocess them into the final working dataset which we use for training, error injection and evaluation.


> [!NOTE]
> The main challange with the original datasets in https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html is that several links are broken.
>
> We wish to use the "resampled" dataset verion for our experiments, however, there exists only the standardized version of this dataset (it's not realistic to inject erronous data into already standardised data as step is usually done closest to the model)


Our strategy to is to take the `processed_endomondoHR_proper_interpolate.npy` (taken from [source website](https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html)) dataset and infer or attach raw data from `endomondoHR.json` (taken from replica in [kaggle](https://www.kaggle.com/datasets/pypiahmad/endomondo-fitness-trajectories))

The final dataset `endomondoHR_proper_interpolated.parquet` should contain all the filtered workouts that are present in `processed_endomondoHR_proper_interpolate.npy` but with the unprocessed features.

> [!NOTE]
> You can download the final dataset from  [here](https://drive.google.com/file/d/1lRjF0MdHsFaMTzYtq5a2opvWvevaZMz9/view?usp=sharing)

## How to run

You can run the master script `dataset_make.sh` or run each script in order:

[1_dataset_download.sh](1_dataset_download.sh) - downloads the initial dataset artifacts from which we construct the final dataset as explained below.
**Output**:
- `.cache/processed_endomondoHR_proper_interpolate.npy`
- `.cache/endomondoHR.json`

---

[2_dataset_preparation.py](./2_dataset_preparation.py) - assembles the final dataset from previously downloaded sources.
**Output**:
- `.cache/endomondoHR_proper_interpolated.parquet`

---

[3_dataset_splitting.py](./3_dataset_splitting.py) - Takes the final dataset and splits it into train/val/test splits using proportional temporal split per user.
**Output**:
- `.cache/train_raw.parquet`
- `.cache/val_raw.parquet`
- `.cache/test_raw.parquet`

---

[4_dataset_preprocessing.py](./4_dataset_preprocessing.py) - Takes the split dataset and standardizes numerical features + encodes the static features ordinally. Besides outputting training-ready data splits, this script also outputs standard scaler + encoder objects (fitted on training set) to be used for preprocessing different versions of val/test datasets.

Please see this [notebook](../../notebooks/test_inference.ipynb) on how to perform inference on unseen test data with all the exported artifacts.

**Output**:
- `.cache/train_preprocessed.parquet`
- `.cache/val_preprocessed.parquet`
- `.cache/test_preprocessed.parquet`
- `.cache/user_standard_scaler.pkl` - [standard scaler object](../../utils/dataset/preprocessing.py) fitted  on training split feature distributions.
- `.cache/static_ordinal_encoder.pkl` - static feature [ordinal encoder](../../utils/dataset/preprocessing.py) fitted on training split feature values.
