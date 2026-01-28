#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEQ_FLOAT_COLS = ["derived_speed", "heart_rate", "altitude", "derived_distance"]
SEQ_INT_COLS = ["timestamp", "time_elapsed"]

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
    - If 99.5% quantile <= 20 => likely m/s (or very conservative km/h)
    - If 99.5% quantile >= 25 => likely km/h
    """
    if speed_vals.size == 0:
        return "unknown"
    q = float(np.quantile(speed_vals, 0.995))
    if q <= 20:
        return "maybe m/s"
    if q >= 25:
        return "maybe km/h"
    return "ambiguous"

def suggest(baseline_path: Path, sample_rows: int, max_gap: int, jump_q: float, max_q: float):
    df = read_parquet_pylist(baseline_path, columns=[c for c in SEQ_FLOAT_COLS if c in pq.read_schema(baseline_path).names])

    out = {}
    for col in [c for c in SEQ_FLOAT_COLS if c in df.columns]:
        vals = flatten(df, col, sample_rows)
        jumps = jump_stats(df, col, sample_rows)

        if vals.size:
            vmax = float(np.quantile(vals, max_q))
        else:
            vmax = float("nan")

        if jumps.size:
            jmax = float(np.quantile(jumps, jump_q))
        else:
            jmax = float("nan")

        out[col] = {"max_q": vmax, "jump_q": jmax, "n": int(vals.size)}

    speed_unit = infer_speed_unit(flatten(df, "derived_speed", sample_rows)) if "derived_speed" in df.columns else "unknown"

    print("\n=== Suggested cleaning parameters (from baseline quantiles) ===")
    print("Baseline:", baseline_path)
    print(f"Sample workouts: {sample_rows}")
    print(f"Inferred speed unit: {speed_unit}")
    print()

    # Recommend values (conservative multipliers)
    def rec_max(x, mult=1.15):
        return None if not np.isfinite(x) else float(x * mult)

    def rec_jump(x, mult=1.5):
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

    print(f"\nmax-gap (gap fill)  ~ {max_gap}  (you chose this; adjust by how long outages last)")
    print("\nRaw stats used:")
    for col, d in out.items():
        print(f"  {col:16s} n={d['n']:,} max@{max_q:.3f}={d['max_q']:.4f} jump@{jump_q:.3f}={d['jump_q']:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Baseline parquet (uncorrupted) to estimate stats from.")
    ap.add_argument("--sample-rows", type=int, default=400, help="Number of workouts to sample (keeps it fast).")
    ap.add_argument("--max-gap", type=int, default=45, help="Suggested max-gap (you can override).")
    ap.add_argument("--jump-q", type=float, default=0.995, help="Quantile for jump thresholds.")
    ap.add_argument("--max-q", type=float, default=0.999, help="Quantile for max thresholds.")
    args = ap.parse_args()

    suggest(Path(args.baseline), args.sample_rows, args.max_gap, args.jump_q, args.max_q)

if __name__ == "__main__":
    main()
