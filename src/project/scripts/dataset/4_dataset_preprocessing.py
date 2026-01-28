import pickle
from pathlib import Path

import pandas as pd

from project.utils.dataset.preprocessing import (
    StaticFeatureOrdinalEncoder,
    UserStandardScaler,
)
from project.utils.dataset.schemas import PREPROCESSED_BASE_SCHEMA

"""
This notebooks take the output dataset from `scripts/dataset`
and splits it into train/val/test, standardizing the features
along the way using different mean and std for every user;
and ordinally encoding the static features (for embedding lookup later on)
"""

# PATHS
SCRIPT_DIR = Path(__file__).parent.absolute()
CACHE_DIR = SCRIPT_DIR / ".cache"

SOURCE_TRAIN_PATH = CACHE_DIR / "train_raw.parquet"
SOURCE_VAL_PATH = CACHE_DIR / "val_raw.parquet"
SOURCE_TEST_PATH = CACHE_DIR / "test_raw.parquet"

OUTPUT_TRAIN_PATH = CACHE_DIR / "train_preprocessed.parquet"
OUTPUT_VAL_PATH = CACHE_DIR / "val_preprocessed.parquet"
OUTPUT_TEST_PATH = CACHE_DIR / "test_preprocessed.parquet"

OUTPUT_SCALER_PATH = CACHE_DIR / "user_standard_scaler.pkl"
OUTPUT_ORDINAL_ENCODER_PATH = CACHE_DIR / "static_ordinal_encoder.pkl"

# CONFIG
COLUMNS_TO_STANDARDIZE = [
    "time_elapsed",
    "heart_rate",
    "altitude",
    "derived_speed",
    "derived_distance",
]

COLUMNS_TO_ENCODE = [
    "userId",
    "sport",
    "gender",
]


for path in (SOURCE_TRAIN_PATH, SOURCE_VAL_PATH, SOURCE_TEST_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Source dataset not found at {path}. "
            "Please run the dataset_splitting.py script first."
        )

train_df = pd.read_parquet(
    SOURCE_TRAIN_PATH,
    dtype_backend="pyarrow",
)

val_df = pd.read_parquet(
    SOURCE_VAL_PATH,
    dtype_backend="pyarrow",
)

test_df = pd.read_parquet(
    SOURCE_TEST_PATH,
    dtype_backend="pyarrow",
)

##########
# STEP 1 #
##########

print("STEP 1: Standardizing features...")

user_standard_scaler = UserStandardScaler(COLUMNS_TO_STANDARDIZE)
user_standard_scaler.fit(train_df)

train_df = user_standard_scaler.transform(train_df)
val_df = user_standard_scaler.transform(val_df)
test_df = user_standard_scaler.transform(test_df)

# save the standadard scaler
with open(OUTPUT_SCALER_PATH, "wb") as f:
    pickle.dump(user_standard_scaler, f)


##########
# STEP 2 #
##########

print("STEP 2: Encoding static features...")

encoder = StaticFeatureOrdinalEncoder(COLUMNS_TO_ENCODE)
encoder.fit(train_df)

train_df = encoder.transform(train_df)
val_df = encoder.transform(val_df)
test_df = encoder.transform(test_df)

# save the encoder
with open(OUTPUT_ORDINAL_ENCODER_PATH, "wb") as f:
    pickle.dump(encoder, f)

##########
# STEP 3 #
##########

print("STEP 3: Saving preprocessed datasets...")

train_df.to_parquet(
    OUTPUT_TRAIN_PATH,
    schema=PREPROCESSED_BASE_SCHEMA,
    engine="pyarrow",
)

val_df.to_parquet(
    OUTPUT_VAL_PATH,
    schema=PREPROCESSED_BASE_SCHEMA,
    engine="pyarrow",
)


test_df.to_parquet(
    OUTPUT_TEST_PATH,
    schema=PREPROCESSED_BASE_SCHEMA,
    engine="pyarrow",
)
