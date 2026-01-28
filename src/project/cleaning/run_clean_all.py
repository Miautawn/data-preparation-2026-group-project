#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from pathlib import Path

PY = ["uv", "run", "python", "src/project/cleaning/advanced_cleaning.py"]

SPORTS = {
    "biking":  dict(speed_max=60, speed_jump=15, hr_jump=50, alt_jump=20, max_gap=60, ema=0.12),
    "running": dict(speed_max=60, speed_jump=26, hr_jump=26, alt_jump=15, max_gap=45, ema=0.12),
    "walking": dict(speed_max=31, speed_jump=16, hr_jump=24, alt_jump=12, max_gap=30, ema=0.15),
}

def run_one(sport: str):
    base = Path("src/project/temp") / sport
    inp = base / f"{sport}_test_raw_corrupted.parquet"
    fit = base / f"{sport}_test_raw.parquet"
    out = base / f"{sport}_test_raw_corrupted_cleaned.parquet"

    if not inp.exists():
        raise FileNotFoundError(inp)
    if not fit.exists():
        raise FileNotFoundError(fit)

    p = SPORTS[sport]
    cmd = PY + [
        "--input", str(inp),
        "--fit-on", str(fit),
        "--output", str(out),
        "--speed-max", str(p["speed_max"]),
        "--speed-max-jump", str(p["speed_jump"]),
        "--hr-max-jump", str(p["hr_jump"]),
        "--altitude-max-jump", str(p["alt_jump"]),
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
