import duckdb
import pandas as pd
import numpy as np
import math
import random
import os

ORIGINAL_FILE = ""
ERRONOUS_FILE = ""
CLEANED_FILE = ""

LOW = 0.10
MID = 0.20
HIGH = 0.30

"""
Creates errors by dropping certain amount of digits of coordinates
percentage depends on lattitude value as GPS gets less accurate
on very high - very low latitudes.
"""

# --- 1. Derivation Logic  ---

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, lambda1 = math.radians(lat1), math.radians(lon1)
    phi2, lambda2 = math.radians(lat2), math.radians(lon2)

    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1

    # Haversine formula
    hav_theta = (math.sin(dphi / 2) ** 2) * (math.cos(dlambda / 2) ** 2) + (
        math.cos((phi1 + phi2) / 2) ** 2
    ) * (math.sin(dlambda / 2) ** 2)
    c = 2 * math.asin(math.sqrt(hav_theta))

    # Radius of Earth in kilometers
    return 6371.0 * c


def _derive_distance(row: pd.Series):
    lat = row["latitude"]
    lon = row["longitude"]

    # Starting point has zero distance travelled
    derived_distances = [0.0]

    # Iterate through the array to calculate distance from previous point
    for i in range(1, len(lat)):
        derived_distances.append(_haversine(lat[i - 1], lon[i - 1], lat[i], lon[i]))

    return np.array(derived_distances)


def _derive_speed(row: pd.Series):
    # timestamps are in seconds, so we get the difference in seconds
    timestamp_diff = np.diff(row["timestamp"]).astype(np.float32)
    timestamp_diff = np.insert(timestamp_diff, 0, 1e-6)  # avoid division by zero for the first point

    # Calculate speed (km/h)
    speed = (row["derived_distance"] / timestamp_diff) * 3600

    return speed


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derives distance and speed from latitude and longitude"""
    df["derived_distance"] = df.apply(_derive_distance, axis=1)
    
    df["derived_speed"] = df.apply(_derive_speed, axis=1)

    return df

# --- 2. Error Injection Logic  ---

def corrupt_coordinate(val):
    return round(val, 3)

def process_gps_arrays(row):
    lons = row['longitude'] 
    lats = row['latitude']
    
    new_lons = []
    new_lats = []

    for lon, lat in zip(lons, lats):

        lat = abs(float(lat))

        if 55 < lat < 70:
            threshold = MID 
        elif lat > 70:
            threshold = HIGH 
        else:
            threshold = LOW 

        if random.random() < threshold:
            new_lons.append(corrupt_coordinate(lon))
            new_lats.append(corrupt_coordinate(lat))
        else:
            new_lons.append(lon)
            new_lats.append(lat)
            
    return new_lons, new_lats

def inject_errors_and_derive():
    print("--- Starting Pipeline ---")
    con = duckdb.connect()
    
    if not os.path.exists(ORIGINAL_FILE):
        print(f"Error: {ORIGINAL_FILE} not found.")
        return

    # 1. Load Data
    df = con.query(f"SELECT * FROM '{ORIGINAL_FILE}'").df()
    
    # 2. Apply Corruption Logic
    processed_series = df.apply(process_gps_arrays, axis=1)
    df['longitude'], df['latitude'] = zip(*processed_series)

    # 3. Re-derive Features
    df = derive_features(df)

    # 4. Write Data
    con.execute("CREATE OR REPLACE TABLE processed_data AS SELECT * FROM df")
    con.execute(f"COPY processed_data TO '{ERRONOUS_FILE}' (FORMAT PARQUET)")
    print("Pipeline Complete.\n")


# --- 3. Cleaning Logic (Implemented) ---

def _smooth_coordinates(row, window_size=5):
    """
    Applies a moving average filter ONLY if the original value has <= 5 decimals.
    """
    lats = pd.Series(row['latitude'])
    lons = pd.Series(row['longitude'])
    
    # 1. Calculate the candidate smoothed values (Rolling Mean)
    smooth_lats = lats.rolling(window=window_size, min_periods=1, center=True).mean()
    smooth_lons = lons.rolling(window=window_size, min_periods=1, center=True).mean()
    
    # 2. Define logic to selectively replace values
    def get_clean_val(original, smoothed):
        # Convert to string to count decimal places
        s_orig = str(original)
        decimals = len(s_orig.split('.')[1]) if '.' in s_orig else 0
        
        # Only replace if precision is significantly lost (5 or less digits)
        if decimals <= 5:
            return round(smoothed, 7)
        
        # Otherwise, keep the original high-quality data
        return original

    # 3. Apply logic list-wise
    clean_lats = [get_clean_val(o, s) for o, s in zip(lats, smooth_lats)]
    clean_lons = [get_clean_val(o, s) for o, s in zip(lons, smooth_lons)]
    
    return clean_lons, clean_lats

def clean_data():
    """
    Reads the ERRONOUS_FILE, cleans the coordinates using a moving average,
    re-derives features, and saves to CLEANED_FILE.
    """
    con = duckdb.connect()

    if not os.path.exists(ERRONOUS_FILE):
        print(f"Error: {ERRONOUS_FILE} not found. Please run inject_errors_and_derive first.")
        return

    # 1. Load Erroneous Data
    df = con.query(f"SELECT * FROM '{ERRONOUS_FILE}'").df()

    # 2. Clean Coordinates
    # Apply smoothing function with a window of 5
    cleaned_series = df.apply(_smooth_coordinates, args=(5,), axis=1)
    
    # Unpack results into the dataframe columns
    df['longitude'], df['latitude'] = zip(*cleaned_series)

    # 3. Re-derive Features (Speed/Distance) based on CLEANED coordinates
    df = derive_features(df)

    # 4. Write Data
    con.execute("CREATE OR REPLACE TABLE cleaned_data AS SELECT * FROM df")
    con.execute(f"COPY cleaned_data TO '{CLEANED_FILE}' (FORMAT PARQUET)")



def debug_print():
    corr = duckdb.query(f"SELECT * FROM '{ORIGINAL_FILE}' LIMIT 1").df()
    err = duckdb.query(f"SELECT * FROM '{ERRONOUS_FILE}' LIMIT 1").df()
    cleaned = duckdb.query(f"SELECT * FROM '{CLEANED_FILE}' LIMIT 1").df()

    files = [corr, err, cleaned]

    for i in range(corr.size):
        orig = float(files[0]['derived_distance'].iloc[0][i])
        err = float(files[1]['derived_distance'].iloc[0][i])
        cle = float(files[2]['derived_distance'].iloc[0][i])
        
        clean = error = 0.0
        clean = clean + abs(orig-cle)
        error = error + abs(orig-err)
    
    print("Difference in distance in erronous:{} vs cleaned:{}".format(error, clean))


if __name__ == "__main__":
    # Only tested on walking
    base = "src\\project\\temp\\walking\\"
    ORIGINAL_FILE = base + "walking_test_raw.parquet"

    for i in range(4):
        ERRONOUS_FILE = base + "ERRONOUS_walking_test_raw{}.parquet".format(str(i))
        CLEANED_FILE = base + "CLEANED_walking_test_raw{}.parquet".format(str(i))


        # Error percentages across different latitude values
        LOW = LOW + i*0.08
        MID = MID + i*0.08
        HIGH = HIGH + i*0.08
        print(LOW, MID, HIGH, "----------------------------------------------------------------------------------------------")

        inject_errors_and_derive()
        clean_data()
        #debug_print()