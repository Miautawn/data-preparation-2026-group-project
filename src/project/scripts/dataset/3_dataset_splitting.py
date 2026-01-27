from pathlib import Path

import pandas as pd

from project.utils.dataset.schemas import BASE_SCHEMA

SCRIPT_DIR = Path(__file__).parent.absolute()
CACHE_DIR = SCRIPT_DIR / ".cache"

SOURCE_PATH = CACHE_DIR / "endomondoHR_proper_interpolated.parquet"
OUTPUT_TRAIN_PATH = CACHE_DIR / "train_raw.parquet"
OUTPUT_VAL_PATH = CACHE_DIR / "val_raw.parquet"
OUTPUT_TEST_PATH = CACHE_DIR / "test_raw.parquet"

if not SOURCE_PATH.exists():
    raise FileNotFoundError(
        f"Source dataset not found at {SOURCE_PATH}. "
        "Please run the dataset_preparation.py script first."
    )

df = pd.read_parquet(
    SOURCE_PATH,
    engine="pyarrow",
    dtype_backend="pyarrow",
)


##########
# STEP 1 #
##########

print("STEP 1: splitting dataset into train/val/test....")

# split into train/val/test
# 70/10/20
# We have to do proportional temporal split per user.

# 1. Ensure data is sorted by User and Time (CRITICAL)
df["first_timestamp"] = df["timestamp"].apply(lambda x: x[0])
df = df.sort_values(by=["first_timestamp", "userId"])

# 2. Calculate the position of each row relative to its user group
user_id_groupby = df.groupby("userId")

# cumcount starts at 0, so row 1 is index 0
df["user_row"] = user_id_groupby.cumcount() + 1
df["user_rows_total"] = user_id_groupby["userId"].transform("count")

df["user_rows_val_start"] = (df["user_rows_total"] * 0.7).astype(int)
df["user_rows_test_start"] = (df["user_rows_total"] * 0.8).astype(int)

train_df = df[df["user_row"] < df["user_rows_val_start"]].copy()
val_df = df[
    (df["user_row"] >= df["user_rows_val_start"])
    & (df["user_row"] < df["user_rows_test_start"])
].copy()
test_df = df[df["user_row"] >= df["user_rows_test_start"]].copy()

##########
# STEP 2 #
##########

print("STEP 2: Exporting train/val/test splits....")

# Drop intermediate columns used for splitting
# and export the results

intermediate_columns = [
    "first_timestamp",
    "user_row",
    "user_rows_total",
    "user_rows_val_start",
    "user_rows_test_start",
]

train_df = train_df.reset_index(drop=True).drop(columns=intermediate_columns)
val_df = val_df.reset_index(drop=True).drop(columns=intermediate_columns)
test_df = test_df.reset_index(drop=True).drop(columns=intermediate_columns)

train_df.to_parquet(
    OUTPUT_TRAIN_PATH,
    schema=BASE_SCHEMA,
    engine="pyarrow",
)


val_df.to_parquet(
    OUTPUT_VAL_PATH,
    schema=BASE_SCHEMA,
    engine="pyarrow",
)


test_df.to_parquet(
    OUTPUT_TEST_PATH,
    schema=BASE_SCHEMA,
    engine="pyarrow",
)
