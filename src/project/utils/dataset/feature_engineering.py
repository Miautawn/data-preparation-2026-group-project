import math

import numpy as np
import pandas as pd


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

    for i in range(1, len(lat)):
        derived_distances.append(_haversine(lat[i - 1], lon[i - 1], lat[i], lon[i]))

    return np.array(derived_distances)


def _derive_speed(row: pd.Series):
    # timestamps are in seconds, so we get the difference in seconds
    timestamp_diff = np.diff(row["timestamp"]).astype(np.float32)
    timestamp_diff = np.insert(timestamp_diff, 0, 1e-6)  # avoid division by zero

    speed = (row["derived_distance"] / timestamp_diff) * 3600

    return speed


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derives distance and speed from latitude and longitude

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: dataframe with 'derived_distance' and 'derived_speed' columns
    """

    df["derived_distance"] = df.apply(_derive_distance, axis=1)
    df["derived_speed"] = df.apply(_derive_speed, axis=1)

    return df
