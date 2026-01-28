#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

PY = ["uv", "run", "python", "src/project/cleaning/advanced_cleaning.py"]

# GPS-focused params (degrees/sample)
# Conservative, realistic: removes teleport spikes while keeping trajectory.
SPORTS = {
    "biking":  dict(gps_max_jump=0.5, max_gap=5, ema=0.3)
    #"running": dict(gps_max_jump=0.02, max_gap=5, ema=0.3)
    #"walking": dict(gps_max_jump=0.01, max_gap=10, ema=0.25),
}

def run_one(sport: str):
    base = Path("src/project/temp") / sport
    inp = base / f"{sport}_test_raw_corrupted.parquet"
    fit = base / f"{sport}_test_raw.parquet"
    out = base / f"{sport}_test_raw_corrupted_cleaned.parquet"

    if not inp.exists():
        raise FileNotFoundError(f"Missing input: {inp}")
    if not fit.exists():
        raise FileNotFoundError(f"Missing baseline (fit-on): {fit}")

    p = SPORTS[sport]

    cmd = PY + [
        "--input", str(inp),
        "--fit-on", str(fit),
        "--output", str(out),

        # GPS-only cleaning (latitude/longitude)
        "--gps-only",
        "--gps-max-jump", str(p["gps_max_jump"]),

        # recovery + smoothing
        "--max-gap", str(p["max_gap"]),
        "--ema-alpha", str(p["ema"]),
    ]

    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    for sport in ["biking", "running", "walking"]:
        run_one(sport)

if __name__ == "__main__":
    main()
