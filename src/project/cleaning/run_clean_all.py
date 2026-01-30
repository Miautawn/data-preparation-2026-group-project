#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

# Command prefix
PY = ["uv", "run", "python", "src/project/cleaning/advanced_cleaning.py"]

# Sport-specific cleaning parameters
SPORTS_CONFIG = {
    "biking":  {"gps_max_jump": 0.5, "max_gap": 5, "ema": 0.3},
    "running": {"gps_max_jump": 0.02, "max_gap": 5, "ema": 0.3},
    "walking": {"gps_max_jump": 0.01, "max_gap": 10, "ema": 0.25},
}

# Added 0 to ensure the baseline itself gets cleaned
SCALES = [0, 1, 2, 4, 8,]

def run_cleaning(sport: str, scale: int):
    # Base directory: data/{sport}
    base = Path("data") / sport
    
    # Baseline path (used for both --fit-on and as input if scale == 0)
    fit_baseline = base / f"{sport}_test_raw.parquet"
    
    # 1. Determine Input and Output paths based on the scale
    if scale == 0:
        inp = fit_baseline # The baseline is the input for Scale 0
        out = base / f"{sport}_test_raw_cleaned.parquet"
    else:
        inp = base / f"erroneous_scale_{scale}_{sport}_data.parquet"
        out = base / f"erroneous_scale_{scale}_{sport}_data_cleaned.parquet"

    # 2. Skip if output already exists
    if out.exists():
        print(f"⏩ Skipping {sport} (Scale {scale}): Output already exists at {out.name}")
        return

    # 3. Skip if input or baseline reference is missing
    if not inp.exists():
        print(f"⚠️  Skipping {sport} (Scale {scale}): Input file not found ({inp.name})")
        return
    if not fit_baseline.exists():
        print(f"⚠️  Skipping {sport} (Scale {scale}): Baseline reference not found ({fit_baseline.name})")
        return

    # 4. Construct Command
    p = SPORTS_CONFIG[sport]
    cmd = PY + [
        "--input", str(inp),
        "--fit-on", str(fit_baseline),
        "--output", str(out),
        "--gps-only",
        "--gps-max-jump", str(p["gps_max_jump"]),
        "--max-gap", str(p["max_gap"]),
        "--ema-alpha", str(p["ema"]),
    ]

    # 5. Execute
    print(f"\n🧹 Cleaning {sport} (Scale {scale})...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Finished: {out.name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error cleaning {sport} (Scale {scale}): {e}")

def main():
    for sport in SPORTS_CONFIG.keys():
        for scale in SCALES:
            run_cleaning(sport, scale)

if __name__ == "__main__":
    main()