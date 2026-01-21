# Dataset generation scripts
These scripts allow you to download the source data and preprocess them into the final working dataset which we use for error injection and evaluation.


> [!NOTE]
> The main challange with the original datasets in https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html is that several links are broken.
>
> We wish to use the "resampled" dataset verion for our experiments, however, there exists only the standardized version of this dataset (it's not realistic to inject erronous data into already standardised data as step is usually done closest to the model)


Our strategy to is to take the `processed_endomondoHR_proper_interpolate.npy` (taken from [source website](https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html)) dataset and infer or attach raw data from `endomondoHR.json` (taken from replica in [kaggle](https://www.kaggle.com/datasets/pypiahmad/endomondo-fitness-trajectories))

The final dataset `endomondoHR_proper_interpolated.parquet` should contain all the filtered workouts that are present in `processed_endomondoHR_proper_interpolate.npy` but with the unprocessed features.

## How to run

You can run the master script `dataset_make.sh`, which will output the final dataset `endomondoHR_proper_interpolated.parquet` in the `.cache` temporary directory.
Alternatively, you can download the final dataset from [here](https://drive.google.com/file/d/1a4UuYh1Oa6ji-anb1OxJa3i78I39R0Y8/view?usp=sharing)

Otherwise, you can run scripts individually

1. Run the dataset download script
```sh
./dataset_download.sh
```

1. Run the dataset preprocessing script:
```sh
uv run python dataset_preprocessing.py
```
