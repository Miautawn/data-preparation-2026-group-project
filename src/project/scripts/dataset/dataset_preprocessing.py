import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.absolute()
CACHE_DIR = SCRIPT_DIR / ".cache"

# final schema must be:
# userId
# id
# sport
# gender
# timestamp
# derived_speed - unnormalized (just take tar_derived_speed)
# heart_rate - unnormalized (just take tar_heart_rate)
# derived_distance - just do derived_speed * (t - t) / 3600
# longitude
# latitude
# altitude -  unnormalized (take from the endomondoHR_proper.json)
# time_elapsed - static sequence across points (calculate natively)
# since_begin - single static number across points (can discard)
# since_last - single static number across points (can discard)


##########
# STEP 1 #
##########

# Read the processed_endomondoHR_proper_interpolate.npy and take as many unprocessed features as possible.
# Export the resulting dataset as parquet using arrow format

print("STEP 1: parsing processed_endomondoHR_proper_interpolate.npy....")

processed_interpolated_data = np.load(
    CACHE_DIR / "processed_endomondoHR_proper_interpolate.npy", allow_pickle=True
)[0]

pure_data_list = []
for i in tqdm(range(processed_interpolated_data.shape[0])):
    row_dict = processed_interpolated_data[i]
    pure_data = {}

    pure_data["id"] = row_dict["id"]
    pure_data["userId"] = row_dict["userId"]
    pure_data["sport"] = row_dict["sport"]
    pure_data["gender"] = row_dict["gender"]

    pure_data["timestamp"] = np.array(row_dict["timestamp"])
    pure_data["derived_speed"] = np.array(row_dict["tar_derived_speed"])
    pure_data["heart_rate"] = np.array(row_dict["tar_heart_rate"])
    pure_data["longitude"] = np.array(row_dict["longitude"])
    pure_data["latitude"] = np.array(row_dict["latitude"])

    timestamp_series = pd.Series(pure_data["timestamp"])
    timestamp_diff = timestamp_series.diff(1).fillna(0)

    # According to the paper: https://dl.acm.org/doi/fullHtml/10.1145/3308558.3313643
    # "Hence we further introduce two derived sequences:
    #   (1) derived distance: Calculate the distance between two data points given their latitudes and longitudes via the Haversine formula;
    #   (2) derived speed: Divide the derived distance and the time interval between two data points."
    # So here we just derive distance backwards from speed and time between points
    pure_data["derived_distance"] = pure_data["derived_speed"] * (
        timestamp_diff.values / 3600
    )
    pure_data["time_elapsed"] = timestamp_diff.cumsum().values.astype(int)

    pure_data_list.append(pure_data)

schema = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("userId", pa.int64()),
        pa.field("sport", pa.string()),
        pa.field("gender", pa.string()),
        pa.field("timestamp", pa.list_(pa.int64())),
        pa.field("derived_speed", pa.list_(pa.float64())),
        pa.field("heart_rate", pa.list_(pa.float64())),
        pa.field("longitude", pa.list_(pa.float64())),
        pa.field("latitude", pa.list_(pa.float64())),
        pa.field("derived_distance", pa.list_(pa.float64())),
        pa.field("time_elapsed", pa.list_(pa.int64())),
    ]
)


table = pa.Table.from_pylist(pure_data_list, schema=schema)
pq.write_table(table, CACHE_DIR / "endomondoHR_proper_interpolated.parquet")

##########
# STEP 2 #
##########

# Read the endomondoHR_proper.json and construct a mapping for prefiltered workouts' timestamp -> altitude
# for faster loopkup and joining to the processed_endomondoHR_proper_interpolate

print("STEP 2: Constructing raw altitude mapping from endomondoHR_proper.json....")

df = pd.read_parquet(
    CACHE_DIR / "endomondoHR_proper_interpolated.parquet",
    dtype_backend="pyarrow",
)

workout_id_set = set(df["id"].values)


# now read the endomondoHR_proper.json line by line and make a mapping for each workout
# for each timestamp -> altitude

workout_timestamp_altitude_mapping = {}
i = 0
with open("./.cache/endomondoHR_proper.json", "r") as f:
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

# Export the final result

print("STEP 4: Exporting final dataframe: 'endomondoHR_proper_interpolated.parquet'")

final_schema = schema.append(pa.field("altitude", pa.list_(pa.float64())))

df.to_parquet(
    CACHE_DIR / "endomondoHR_proper_interpolated.parquet",
    schema=final_schema,
    engine="pyarrow",
)
