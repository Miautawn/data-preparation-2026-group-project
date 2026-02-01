#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Include GPS columns now
SEQ_FLOAT_COLS = ["derived_speed", "heart_rate", "altitude", "derived_distance", "latitude", "longitude"]


def read_parquet_pylist(path: Path, columns=None) -> pd.DataFrame:
    t = pq.read_table(path, columns=columns, use_pandas_metadata=False)
    return pd.DataFrame({c: t[c].to_pylist() for c in t.column_names})


def flatten(df: pd.DataFrame, col: str, max_rows: int) -> np.ndarray:
    seqs = df[col].tolist()[:max_rows]
    parts = []
    for s in seqs:
        a = np.asarray(s, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            parts.append(a)
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def jump_stats(df: pd.DataFrame, col: str, max_rows: int) -> np.ndarray:
    seqs = df[col].tolist()[:max_rows]
    parts = []
    for s in seqs:
        a = np.asarray(s, dtype=float)
        a = a[np.isfinite(a)]
        if a.size >= 2:
            d = np.abs(np.diff(a))
            if d.size:
                parts.append(d)
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def infer_speed_unit(speed_vals: np.ndarray) -> str:
    """
    Very rough heuristic:
    - If 99.5% quantile <= 20 => maybe m/s (or very conservative km/h)
    - If 99.5% quantile >= 25 => maybe km/h
    """
    if speed_vals.size == 0:
        return "unknown"
    q = float(np.quantile(speed_vals, 0.995))
    if q <= 20:
        return "maybe m/s"
    if q >= 25:
        return "maybe km/h"
    return "ambiguous"


def degrees_to_meters_lat(dlat_deg: float) -> float:
    # ~111.32 km per degree latitude
    return float(abs(dlat_deg) * 111_320.0)


def degrees_to_meters_lon(dlon_deg: float, at_lat_deg: float) -> float:
    # meters per degree longitude shrinks by cos(latitude)
    return float(abs(dlon_deg) * 111_320.0 * np.cos(np.deg2rad(at_lat_deg)))


def suggest(
    baseline_path: Path,
    sample_rows: int,
    max_gap: int,
    jump_q: float,
    max_q: float,
    gps_jump_q: float,
    gps_mult: float,
) -> None:
    schema_names = pq.read_schema(baseline_path).names
    cols = [c for c in SEQ_FLOAT_COLS if c in schema_names]
    df = read_parquet_pylist(baseline_path, columns=cols)

    out = {}
    for col in cols:
        vals = flatten(df, col, sample_rows)
        jumps = jump_stats(df, col, sample_rows)

        vmax = float(np.quantile(vals, max_q)) if vals.size else float("nan")
        jmax = float(np.quantile(jumps, jump_q)) if jumps.size else float("nan")

        out[col] = {"max_q": vmax, "jump_q": jmax, "n": int(vals.size)}

    speed_unit = infer_speed_unit(flatten(df, "derived_speed", sample_rows)) if "derived_speed" in df.columns else "unknown"

    print("\n=== Suggested cleaning parameters (from baseline quantiles) ===")
    print("Baseline:", baseline_path)
    print(f"Sample workouts: {sample_rows}")
    print(f"Inferred speed unit: {speed_unit}")
    print()

    # Recommend values (conservative multipliers)
    def rec_max(x: float, mult: float = 1.15):
        return None if not np.isfinite(x) else float(x * mult)

    def rec_jump(x: float, mult: float = 1.50):
        return None if not np.isfinite(x) else float(x * mult)

    # Speed
    if "derived_speed" in out:
        smax = rec_max(out["derived_speed"]["max_q"], mult=1.20)
        sjmp = rec_jump(out["derived_speed"]["jump_q"], mult=1.70)
        print("derived_speed:")
        print(f"  speed-max        ~ {smax:.2f}" if smax is not None else "  speed-max        ~ (n/a)")
        print(f"  speed-max-jump   ~ {sjmp:.2f}" if sjmp is not None else "  speed-max-jump   ~ (n/a)")

    # Heart rate
    if "heart_rate" in out:
        hjmp = rec_jump(out["heart_rate"]["jump_q"], mult=1.40)
        print("heart_rate:")
        print("  hr-min / hr-max  ~ 30 / 230  (keep default unless you have reason)")
        print(f"  hr-max-jump      ~ {hjmp:.2f}" if hjmp is not None else "  hr-max-jump      ~ (n/a)")

    # Altitude
    if "altitude" in out:
        ajmp = rec_jump(out["altitude"]["jump_q"], mult=1.60)
        print("altitude:")
        print(f"  altitude-max-jump ~ {ajmp:.2f}" if ajmp is not None else "  altitude-max-jump ~ (n/a)")

    # -------------------------
    # GPS: recommend gps-max-jump
    # -------------------------
    if "latitude" in df.columns or "longitude" in df.columns:
        lat_j = jump_stats(df, "latitude", sample_rows) if "latitude" in df.columns else np.asarray([], dtype=float)
        lon_j = jump_stats(df, "longitude", sample_rows) if "longitude" in df.columns else np.asarray([], dtype=float)

        lat_q = float(np.quantile(lat_j, gps_jump_q)) if lat_j.size else float("nan")
        lon_q = float(np.quantile(lon_j, gps_jump_q)) if lon_j.size else float("nan")

        # Use the larger of (lat, lon) quantiles as a conservative baseline
        base = np.nanmax([lat_q, lon_q])
        gps_max_jump = float(base * gps_mult) if np.isfinite(base) else float("nan")

        print("\nGPS (latitude/longitude):")
        if np.isfinite(gps_max_jump):
            print(f"  gps-max-jump     ~ {gps_max_jump:.7f} degrees/sample  (based on q={gps_jump_q} * {gps_mult})")
        else:
            print("  gps-max-jump     ~ (n/a)")

        # Provide a rough meters/sample interpretation for sanity checking.
        # Use median latitude for lon conversion if possible.
        if np.isfinite(gps_max_jump):
            lat_vals = flatten(df, "latitude", sample_rows) if "latitude" in df.columns else np.asarray([], dtype=float)
            ref_lat = float(np.median(lat_vals)) if lat_vals.size else 0.0

            m_lat = degrees_to_meters_lat(gps_max_jump)
            m_lon = degrees_to_meters_lon(gps_max_jump, ref_lat)
            print(f"  approx distance  ~ {m_lat:.1f} m/sample (lat), ~ {m_lon:.1f} m/sample (lon at lat≈{ref_lat:.2f}°)")

        # Also print raw jump quantiles
        if np.isfinite(lat_q):
            print(f"  raw lat jump@{gps_jump_q:.3f} = {lat_q:.7f} deg")
        if np.isfinite(lon_q):
            print(f"  raw lon jump@{gps_jump_q:.3f} = {lon_q:.7f} deg")

    print(f"\nmax-gap (gap fill)  ~ {max_gap}  (you chose this; adjust by how long outages last)")
    print("\nRaw stats used:")
    for col, d in out.items():
        print(
            f"  {col:16s} n={d['n']:,} "
            f"max@{max_q:.3f}={d['max_q']:.6f} "
            f"jump@{jump_q:.3f}={d['jump_q']:.6f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Baseline parquet (uncorrupted) to estimate stats from.")
    ap.add_argument("--sample-rows", type=int, default=400, help="Number of workouts to sample (keeps it fast).")

    ap.add_argument("--max-gap", type=int, default=45, help="Suggested max-gap (you can override).")

    ap.add_argument("--jump-q", type=float, default=0.995, help="Quantile for jump thresholds (speed/hr/alt).")
    ap.add_argument("--max-q", type=float, default=0.999, help="Quantile for max thresholds (speed).")

    # GPS-specific knobs
    ap.add_argument("--gps-jump-q", type=float, default=0.999, help="Quantile for GPS jump threshold recommendation.")
    ap.add_argument("--gps-mult", type=float, default=1.30, help="Multiplier applied to GPS jump quantile.")

    args = ap.parse_args()

    suggest(
        baseline_path=Path(args.baseline),
        sample_rows=args.sample_rows,
        max_gap=args.max_gap,
        jump_q=args.jump_q,
        max_q=args.max_q,
        gps_jump_q=args.gps_jump_q,
        gps_mult=args.gps_mult,
    )


if __name__ == "__main__":
    main()
