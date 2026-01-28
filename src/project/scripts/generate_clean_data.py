#!/usr/bin/env python3
"""
Script to generate CLEANED datasets for FitRec.
It produces:
1. cleaned_scale_0_{sport}.parquet  (Cleaned Baseline)
2. cleaned_scale_X_{sport}.parquet  (Cleaned version of Scale 2, 5, 10)

Workflow:
- Check if output exists (SKIP if yes)
- Load Erroneous File
- Swap 'latitude'/'longitude' with 'erroneous_latitude'/'erroneous_longitude'
- Apply GenericSequenceCleaner (Smoothing + Clipping)
- Recalculate 'derived_speed' and 'derived_distance' from the new coordinates
- Save
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path


# ---- 1. Path Setup ----
def add_src_to_path():
    p = Path.cwd().resolve()
    for parent in [p] + list(p.parents):
        if (parent / "src").exists():
            sys.path.insert(0, str(parent / "src"))
            return parent / "data"
    raise RuntimeError("Could not find 'src' or 'data' folder.")


try:
    DATA_ROOT = add_src_to_path()
except RuntimeError as e:
    print(f"❌ Setup Error: {e}")
    sys.exit(1)

from project.cleaning.cleaning_generic import (
    GenericSequenceCleaner,
    CleaningConfig,
    _read_parquet_pylist,
)


# ---- 2. Physics Recalculation Logic ----
def derive_features_for_row(lats, lons, timestamps):
    """
    Recalculates speed and cumulative distance from lists of coordinates/times.
    """
    if len(lats) < 2:
        return [], []

    lats = np.array(lats, dtype=float)
    lons = np.array(lons, dtype=float)
    times = np.array(timestamps, dtype=float)

    # 1. Calculate Distance between points (Haversine approx or simple Euclidean)
    # Using simple Euclidean with approx conversion for speed (sufficient for features)
    # 1 deg lat ~ 111.32 km
    d_lat = np.diff(lats)
    d_lon = np.diff(lons)

    # Simple approx for small distances: sqrt(dx^2 + dy^2)
    # Note: A more precise haversine is better, but this matches typical FitRec prep
    dist_m = np.sqrt(
        (d_lat * 111320) ** 2 + (d_lon * 111320 * np.cos(np.radians(lats[:-1]))) ** 2
    )

    # 2. Calculate Speed
    dt = np.diff(times)
    dt[dt <= 0] = 1.0  # Avoid division by zero
    speed_mps = dist_m / dt

    # Pad first element to match length
    speed_mps = np.concatenate(([0.0], speed_mps))

    # 3. Cumulative Distance
    cum_dist = np.concatenate(([0.0], np.cumsum(dist_m)))

    return speed_mps.tolist(), cum_dist.tolist()


def recalculate_physics(df):
    """Applies physics recalculation to the whole DataFrame."""
    new_speeds = []
    new_dists = []

    for _, row in df.iterrows():
        s, d = derive_features_for_row(
            row["latitude"], row["longitude"], row["timestamp"]
        )
        new_speeds.append(s)
        new_dists.append(d)

    df["derived_speed"] = new_speeds
    df["derived_distance"] = new_dists
    return df


# ---- 3. Main Processing Logic ----

# Strict Configuration to handle Scale 10 Noise
STRICT_CONFIG = CleaningConfig(
    smooth=True,
    smooth_window=11,  # Strong smoothing
    clip=True,
    clip_q_high=0.999,
    impute=True,
)

# Manual Bounds (Since Baseline is noisy, we hardcode human limits)
HUMAN_BOUNDS = {
    "derived_speed": (0.0, 20.0),  # Max 20 m/s (72 km/h) - Generous for biking
    "heart_rate": (30.0, 220.0),
    "altitude": (-500.0, 9000.0),
    "derived_distance": (0.0, 500000.0),  # Max 500km ride
}


def process_sport(sport):
    print(f"\n🚴🏃🚶 Processing Sport: {sport.upper()}")
    sport_dir = DATA_ROOT / sport
    baseline_path = sport_dir / f"{sport}_test_raw.parquet"

    if not baseline_path.exists():
        print(f"  ❌ Baseline not found: {baseline_path}")
        return

    # 1. Setup Cleaner
    # We fit on baseline primarily to get lat/lon bounds, but override the rest
    df_baseline = _read_parquet_pylist(baseline_path)
    cleaner = GenericSequenceCleaner(config=STRICT_CONFIG)
    cleaner.fit(df_baseline)

    # Override with human limits
    cleaner.clip_bounds_.update(HUMAN_BOUNDS)
    print("  🧠 Cleaner Configured (Human Bounds Applied)")

    # 2. Process Each Scale
    scales = [0, 2, 5, 10]

    for scale in scales:
        # Define Input/Output
        if scale == 0:
            input_path = baseline_path
            output_name = f"cleaned_scale_0_{sport}.parquet"
        else:
            input_path = sport_dir / f"erroneous_scale_{scale}_{sport}_data.parquet"
            output_name = f"cleaned_scale_{scale}_{sport}.parquet"

        output_path = sport_dir / output_name

        # ---- SKIPPING LOGIC ----
        if output_path.exists():
            print(f"  ⏭️  Skipping {output_name} (Already exists)")
            continue
        # ------------------------

        if not input_path.exists():
            print(f"  ⚠️  Missing Input: {input_path.name}")
            continue

        try:
            # A. Load
            df = _read_parquet_pylist(input_path)

            # B. Swap Erroneous Columns (Crucial Step!)
            if scale > 0:
                if "erroneous_latitude" in df.columns:
                    df["latitude"] = df["erroneous_latitude"]
                    df["longitude"] = df["erroneous_longitude"]
                else:
                    print(
                        f"    ⚠️ Warning: {input_path.name} is missing 'erroneous_' columns. Using existing lat/lon."
                    )

            # C. Clean (Smooth + Clip)
            df_cleaned = cleaner.transform(df)

            # D. Recalculate Physics (Speed/Distance) based on Cleaned Coords
            df_cleaned = recalculate_physics(df_cleaned)

            # E. Save
            df_cleaned.to_parquet(output_path, index=False)
            print(f"  ✅ Generated: {output_name}")

        except Exception as e:
            print(f"  ❌ Failed {output_name}: {e}")


# ---- 4. Execution ----
def main():
    for sport in ["biking", "running", "walking"]:
        process_sport(sport)
    print("\n✨ All cleaned files generated.")


if __name__ == "__main__":
    main()
