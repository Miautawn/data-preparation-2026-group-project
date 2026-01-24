from pathlib import Path

import pandas as pd

from project.utils.modeling import FitRecDataset, FitRecModel, train_model

# PATHS
SCRIPT_DIR = Path(__file__).parent.absolute()
CACHE_DIR = SCRIPT_DIR / ".cache"
DATASET_CACHE_DIR = SCRIPT_DIR.parent / "dataset" / ".cache"

SOURCE_TRAIN_PATH = DATASET_CACHE_DIR / "train_preprocessed.parquet"
SOURCE_VAL_PATH = DATASET_CACHE_DIR / "val_preprocessed.parquet"

OUTPUT_MODEL_PATH = CACHE_DIR / "fitrec_model_non_autoregressive.pt"

# CONFIG
# This flag indicates whether to train the model
# autoregressively using previous heart rate as input
USE_HEARTRATE_INPUT = False

NUMERICAL_INPUTS = [
    "time_elapsed_standardized",
    "altitude_standardized",
    "derived_speed_standardized",
    "derived_distance_standardized",
]
CATEGORICAL_INPUTS = ["userId_idx", "sport_idx", "gender_idx"]

# a standardized heart rate input column is only used if USE_HEARTRATE_INPUT is True
# it is specified separately as unlike other features, it is windowed differently with lag
HEARTRATE_INPUT_COLUMN = "heart_rate_standardized"
HEARTRATE_OUTPUT_COLUMN = "heart_rate"
WORKOUT_ID_COLUMNS = "id"

###
BATCH_SIZE = 2048
EPOCHS = 5
LEARNING_RATE = 0.001
L2_NORM = 0.001
N_WORKERS = 8
###

# SAFETY CHECKS
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True)

for path in (SOURCE_TRAIN_PATH, SOURCE_VAL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Source dataset not found at {path}. "
            "Please run the scripts/dataset/4_dataset_preprocessing.py script first."
        )


##########
# STEP 1 #
##########

print("STEP 1: Reading input data...")

all_columns = (
    NUMERICAL_INPUTS
    + CATEGORICAL_INPUTS
    + [HEARTRATE_INPUT_COLUMN, HEARTRATE_OUTPUT_COLUMN, WORKOUT_ID_COLUMNS]
)

# Can't read only specific columns from parquet for some goofy reason
train_df = pd.read_parquet(
    SOURCE_TRAIN_PATH,
    dtype_backend="pyarrow",
)[all_columns]

val_df = pd.read_parquet(
    SOURCE_VAL_PATH,
    dtype_backend="pyarrow",
)[all_columns]


##########
# STEP 2 #
##########

print("STEP 2: Creating training/val datasets...")

dataset_arguments = {
    "numerical_columns": NUMERICAL_INPUTS,
    "categorical_columns": CATEGORICAL_INPUTS,
    "heartrate_input_column": HEARTRATE_INPUT_COLUMN,
    "heartrate_output_column": HEARTRATE_OUTPUT_COLUMN,
    "workout_id_column": WORKOUT_ID_COLUMNS,
    "use_heartrate_input": USE_HEARTRATE_INPUT,
}

train_dataset = FitRecDataset(
    train_df,
    **dataset_arguments,
)

val_dataset = FitRecDataset(
    val_df,
    **dataset_arguments,
)

##########
# STEP 3 #
##########

print("STEP 3: Instantiating the model...")

model = FitRecModel(
    n_sequential_features=len(NUMERICAL_INPUTS),
    n_users=train_df["userId_idx"].nunique(),
    n_sports=train_df["sport_idx"].nunique(),
    n_genders=train_df["gender_idx"].nunique(),
    use_heartrate_input=USE_HEARTRATE_INPUT,
)

print("STEP 4: Training the model...")

model = train_model(
    model,
    train_dataset,
    val_dataset,
    model_save_path=OUTPUT_MODEL_PATH,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    l2_norm=L2_NORM,
    n_workers=N_WORKERS,
)
