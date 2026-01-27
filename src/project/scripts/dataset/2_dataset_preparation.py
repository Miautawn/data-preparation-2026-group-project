import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from project.utils.dataset import derive_features
from project.utils.dataset.schemas import BASE_SCHEMA

SCRIPT_DIR = Path(__file__).parent.absolute()
CACHE_DIR = SCRIPT_DIR / ".cache"

SOURCE_PATH = CACHE_DIR / "processed_endomondoHR_proper_interpolate.npy"
ALTITUDE_SOURCE_PATH = CACHE_DIR / "endomondoHR_proper.json"
OUTPUT_PATH = CACHE_DIR / "endomondoHR_proper_interpolated.parquet"

if not SOURCE_PATH.exists():
    raise FileNotFoundError(
        f"Source dataset not found at {SOURCE_PATH}. "
        "Please run the dataset_download.sh script first."
    )

##########
# STEP 1 #
##########

# Read the processed_endomondoHR_proper_interpolate.npy and take as many unprocessed features as possible.
# Export the resulting dataset as parquet using arrow format

print("STEP 1: parsing processed_endomondoHR_proper_interpolate.npy....")

processed_interpolated_data = np.load(SOURCE_PATH, allow_pickle=True)[0]

pure_data_list = []
for i in tqdm(range(processed_interpolated_data.shape[0])):
    row_dict = processed_interpolated_data[i]
    pure_data = {}

    pure_data["id"] = row_dict["id"]
    pure_data["userId"] = row_dict["userId"]
    pure_data["sport"] = row_dict["sport"]
    pure_data["gender"] = row_dict["gender"]

    pure_data["timestamp"] = np.array(row_dict["timestamp"])
    pure_data["heart_rate"] = np.array(row_dict["tar_heart_rate"])
    pure_data["longitude"] = np.array(row_dict["longitude"])
    pure_data["latitude"] = np.array(row_dict["latitude"])

    timestamp_series = pd.Series(pure_data["timestamp"])
    timestamp_diff = timestamp_series.diff(1).fillna(0)

    pure_data["derived_speed"] = np.array([0.0])
    pure_data["derived_distance"] = np.array([0.0])

    pure_data["time_elapsed"] = timestamp_diff.cumsum().values.astype(int)

    # Add dummy altitude field to satisfy schema

    pure_data_list.append(pure_data)
    pure_data["altitude"] = np.array([0])

table = pa.Table.from_pylist(pure_data_list, schema=BASE_SCHEMA)
pq.write_table(table, OUTPUT_PATH)

del processed_interpolated_data, pure_data_list, table


##########
# STEP 2 #
##########

df = pd.read_parquet(
    OUTPUT_PATH,
    dtype_backend="pyarrow",
)

# Read the endomondoHR_proper.json and construct a mapping for prefiltered workouts' timestamp -> altitude
# for faster loopkup and joining to the processed_endomondoHR_proper_interpolate

print("STEP 2: Constructing raw altitude mapping from endomondoHR_proper.json....")

workout_id_set = set(df["id"].values)

# now read the endomondoHR_proper.json line by line and make a mapping for each workout
# for each timestamp -> altitude

workout_timestamp_altitude_mapping = {}
i = 0
with open(ALTITUDE_SOURCE_PATH, "r") as f:
    for row in tqdm(f):
        row = json.loads(row.replace("'", '"'))

        workout_id = row["id"]
        if workout_id in workout_id_set:
            workout_timestamp_altitude_mapping[workout_id] = {
                k: v for k, v in zip(row["timestamp"], row["altitude"])
            }

##########
# STEP 3 #
##########

# "Join" the unprocessed altitude from endomondoHR_proper.json to processed_endomondoHR_proper_interpolate using the workout id and timestamp as join keys

print("STEP 3: Attaching raw altitudes...")


def get_altitude_array(workout_id: int, timestamps: list[int]):
    return [workout_timestamp_altitude_mapping[workout_id][t] for t in timestamps]


df["altitude"] = df[["id", "timestamp"]].apply(
    lambda x: get_altitude_array(x["id"], x["timestamp"]), axis=1
)


##########
# STEP 4 #
##########

print("STEP 4: Deriving distance and speed from coordinates...")

# According to the paper: https://dl.acm.org/doi/fullHtml/10.1145/3308558.3313643
# "Hence we further introduce two derived sequences:
#   (1) derived distance: Calculate the distance between two data points given their latitudes and longitudes via the Haversine formula;
#   (2) derived speed: Divide the derived distance and the time interval between two data points."
# Thus, we attempt to derive them in the same fashion
df = derive_features(df)


##########
# STEP 5 #
##########

print("STEP 5: Consolidating sport types...")

# We'll be experimenting and splitting the dataset by 3 views:
# Walking, Running, Biking

# To do this, we consolidate the sport categories into these 3 main categories.
# All other sports will be categorized will be dropped.

sport_category_mapping = {
    "bike": "biking",
    "run": "running",
    "walk": "walking",
    "orienteering": "walking",
    "hiking": "walking",
    "fitness walking": "walking",
}

df["sport"] = df["sport"].map(sport_category_mapping)
df = df[df["sport"].notnull()]

##########
# STEP 6 #
##########

# Filtering out users who have less than 10 workouts - this will ensure enough samples for train/val/test splits
# and match the advertised dataset size of 102,343 workouts
print("STEP 6: Filtering users with less than 10 workouts....")

user_ids = df["userId"].value_counts()
user_ids_mask = user_ids[user_ids >= 10].index

df = df[df["userId"].isin(user_ids_mask)]

##########
# STEP 7 #
##########

# Export the final result

print("STEP 7: Exporting final dataframe: 'endomondoHR_proper_interpolated.parquet'")

df.to_parquet(
    OUTPUT_PATH,
    schema=BASE_SCHEMA,
    engine="pyarrow",
)
