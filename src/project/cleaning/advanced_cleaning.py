#!/usr/bin/env python3
"""
advanced_cleaning.py

A more realistic "smartwatch-style" (causal) cleaner for FitRec-style Endomondo parquet files where
sequential signals are stored as list columns per workout.

Compared to a basic offline cleaner, this implements a causal / forward-only pipeline that:
- never uses future values to clean current time t
- applies plausibility checks (range filters)
- applies spike/glitch detection using causal jump limits
- imputes missing values using forward-fill WITH a configurable max-gap (recovery window)
- uses a causal EMA smoother (and can apply EMA before jump-filter to reduce false spike detection)
- clips using bounds fitted from a baseline/train file (optional but recommended)

It is designed to work with your artifacts layout (walking/running/biking) and to keep column names stable.

Typical usage (single file):
  uv run python src/project/cleaning/advanced_cleaning.py \
    --input  src/project/temp/biking/biking_test_raw_corrupted.parquet \
    --fit-on src/project/temp/biking/biking_test_raw.parquet \
    --output src/project/temp/biking/biking_test_raw_corrupted_cleaned.parquet \
    --mode causal

Directory usage:
  uv run python src/project/cleaning/advanced_cleaning.py \
    --root src/project/temp --pattern "**/*_test_raw_corrupted.parquet" --mode causal

Notes on deletion:
- By default, invalid workouts (length mismatch or too-short sequences) are DROPPED.
  Set --no-drop-invalid to keep them (they'll be cleaned best-effort).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---------------------------
# Defaults / schema
# ---------------------------
SEQ_COLS_FLOAT_DEFAULT = [
    "derived_speed",
    "heart_rate",
    "longitude",
    "latitude",
    "derived_distance",
    "altitude",
]
SEQ_COLS_INT_DEFAULT = ["timestamp", "time_elapsed"]


# ---------------------------
# Robust parquet reader
# ---------------------------
def read_parquet_pylist(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Robustly read parquet with list columns by extracting each column as Python lists.
    Avoids pandas/pyarrow dtype backend issues.
    """
    table = pq.read_table(path, columns=columns, use_pandas_metadata=False)
    return pd.DataFrame({c: table[c].to_pylist() for c in table.column_names})


# ---------------------------
# Basic helpers
# ---------------------------
def _as_list(x) -> list:
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except Exception:
        return []


def _to_float_array(x) -> np.ndarray:
    try:
        return np.asarray(_as_list(x), dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def _to_int_array(x) -> np.ndarray:
    try:
        return np.asarray(_as_list(x), dtype=np.int64)
    except Exception:
        return np.asarray([], dtype=np.int64)


# ---------------------------
# Causal helpers (smartwatch-like)
# ---------------------------
def _ema_causal(y: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA. NaNs produce previous EMA output (or NaN if not initialized)."""
    if y.size == 0 or alpha <= 0:
        return y
    out = y.copy()
    ema = np.nan
    for i in range(out.size):
        x = out[i]
        if not np.isfinite(x):
            out[i] = ema
            continue
        if not np.isfinite(ema):
            ema = x
        else:
            ema = alpha * x + (1.0 - alpha) * ema
        out[i] = ema
    return out


def _jump_filter_causal(y: np.ndarray, max_jump: float) -> np.ndarray:
    """
    Causal spike filter: if current finite value jumps too much vs previous finite value,
    mark current as NaN.
    """
    if y.size == 0 or not np.isfinite(max_jump) or max_jump <= 0:
        return y
    out = y.copy()
    prev = np.nan
    for i in range(out.size):
        x = out[i]
        if not np.isfinite(x):
            continue
        if np.isfinite(prev) and abs(x - prev) > max_jump:
            out[i] = np.nan
            continue
        prev = x
    return out


def _forward_fill_with_max_gap(y: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Forward-fill (LOCF) but only up to max_gap consecutive NaNs.
    If a NaN-run exceeds max_gap, those values remain NaN (so we can fallback more intelligently).
    Leading NaNs remain NaN.

    Example: max_gap=10 means you can bridge up to 10 missing samples, but longer outages remain missing.
    """
    if y.size == 0:
        return y
    out = y.copy()
    last = np.nan
    gap = 0
    for i in range(out.size):
        if np.isfinite(out[i]):
            last = out[i]
            gap = 0
        else:
            gap += 1
            if np.isfinite(last) and gap <= max_gap:
                out[i] = last
            # else: keep NaN
    return out


def _fill_leading_nans(y: np.ndarray, fill_value: float) -> np.ndarray:
    """Fill leading NaNs (only at the start) with fill_value."""
    if y.size == 0:
        return y
    out = y.copy()
    i = 0
    while i < out.size and not np.isfinite(out[i]):
        out[i] = fill_value
        i += 1
    return out


# ---------------------------
# Config
# ---------------------------
@dataclass
class AdvancedCleaningConfig:
    seq_float_cols: List[str] = field(default_factory=lambda: SEQ_COLS_FLOAT_DEFAULT.copy())
    seq_int_cols: List[str] = field(default_factory=lambda: SEQ_COLS_INT_DEFAULT.copy())

    # Validity / dropping
    drop_invalid_length_rows: bool = True
    min_seq_len: int = 5

    # Mode
    mode: str = "causal"  # "causal" recommended; "offline" not implemented here

    # Fit-based clipping
    clip: bool = True
    clip_q_low: float = 0.005
    clip_q_high: float = 0.995

    # Causal imputation + recovery window
    impute: bool = True
    max_gap: int = 15  # allow recovery after short noise bursts without permanent flatline

    # Causal smoothing
    smooth: bool = True
    ema_alpha: float = 0.20
    prefilter_ema: bool = True  # apply a light EMA before jump-filter to reduce false positives
    prefilter_alpha: float = 0.12

    smooth_cols: Optional[List[str]] = None  # defaults to hr + speed + altitude if None

    # Plausibility ranges (wearable-realistic defaults)
    hr_min: float = 30.0
    hr_max: float = 230.0

    # speed is often in m/s in such datasets; adjust if yours is km/h
    speed_min: float = 0.0
    speed_max: float = 12.0

    # jump thresholds (tuneable)
    hr_max_jump: float = 50.0
    speed_max_jump: float = 6.0
    altitude_max_jump: float = 20.0

    # GPS plausibility
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0

    # If a sequence is mostly missing after cleaning, we fallback to robust stats:
    # - first try workout median (from available points in that workout)
    # - else fallback to fitted global median (from fit-on file)
    fallback_to_global_median_if_needed: bool = True


# ---------------------------
# Cleaner
# ---------------------------
class AdvancedSequenceCleaner:
    """
    Smartwatch-style (causal) cleaner. Fit provides global clip bounds + global medians.
    """

    def __init__(self, config: Optional[AdvancedCleaningConfig] = None):
        self.config = config or AdvancedCleaningConfig()
        self.clip_bounds_: Optional[Dict[str, Tuple[float, float]]] = None
        self.global_median_: Optional[Dict[str, float]] = None

        if self.config.smooth_cols is None:
            self.config.smooth_cols = [
                c for c in ["heart_rate", "derived_speed", "altitude"] if c in self.config.seq_float_cols
            ]

    def _validate_schema(self, df: pd.DataFrame) -> None:
        required = ["id"] + self.config.seq_int_cols + self.config.seq_float_cols
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def fit(self, df_fit: pd.DataFrame) -> "AdvancedSequenceCleaner":
        """
        Compute global clip bounds and global medians from df_fit (baseline/train).
        """
        self._validate_schema(df_fit)
        cfg = self.config

        bounds: Dict[str, Tuple[float, float]] = {}
        medians: Dict[str, float] = {}

        for col in cfg.seq_float_cols:
            vals = []
            for x in df_fit[col].tolist():
                arr = _to_float_array(x)
                if arr.size:
                    vals.append(arr)
            if not vals:
                bounds[col] = (-np.inf, np.inf)
                medians[col] = 0.0
                continue

            v = np.concatenate(vals)
            v = v[np.isfinite(v)]
            if v.size == 0:
                bounds[col] = (-np.inf, np.inf)
                medians[col] = 0.0
                continue

            lo = float(np.quantile(v, cfg.clip_q_low))
            hi = float(np.quantile(v, cfg.clip_q_high))
            bounds[col] = (lo, hi)
            medians[col] = float(np.median(v))

        self.clip_bounds_ = bounds
        self.global_median_ = medians
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean all float list-columns causally; keep schema stable.
        """
        self._validate_schema(df)
        if self.clip_bounds_ is None or self.global_median_ is None:
            raise RuntimeError("Cleaner not fitted. Call fit() first (use baseline/train).")

        cfg = self.config
        df_out = df.copy()

        # Ensure list columns are object dtype to allow assignment of Python lists
        for col in cfg.seq_int_cols + cfg.seq_float_cols:
            df_out[col] = df_out[col].astype(object)

        # Validate lengths against timestamp
        ts_lens = df_out["timestamp"].apply(lambda x: len(_as_list(x)))
        valid = ts_lens >= cfg.min_seq_len
        for col in cfg.seq_float_cols + cfg.seq_int_cols:
            valid &= df_out[col].apply(lambda x: len(_as_list(x))).eq(ts_lens)

        if cfg.drop_invalid_length_rows:
            df_out = df_out.loc[valid].reset_index(drop=True)

        def _workout_fallback(y: np.ndarray, col: str) -> float:
            """Fallback fill value for a workout/column."""
            # workout median if any finite values exist
            finite = y[np.isfinite(y)]
            if finite.size:
                return float(np.median(finite))
            # else global median
            return float(self.global_median_.get(col, 0.0))

        def clean_float_seq(x, col: str) -> list:
            y = _to_float_array(x)
            if y.size == 0:
                return []

            # normalize non-finite -> NaN
            y[~np.isfinite(y)] = np.nan

            # -------- plausibility filters --------
            if col == "heart_rate":
                y[(y < cfg.hr_min) | (y > cfg.hr_max)] = np.nan
            elif col == "derived_speed":
                y[(y < cfg.speed_min) | (y > cfg.speed_max)] = np.nan
            elif col == "latitude":
                y[(y < cfg.lat_min) | (y > cfg.lat_max)] = np.nan
            elif col == "longitude":
                y[(y < cfg.lon_min) | (y > cfg.lon_max)] = np.nan
            # altitude / distance: we mostly use jump + clip; keep negatives if dataset uses them (altitude can be below sea level)

            # -------- optional prefilter smoothing (helps reduce false spike flags) --------
            if cfg.prefilter_ema and col in (cfg.smooth_cols or []):
                y = _ema_causal(y, cfg.prefilter_alpha)

            # -------- spike/jump filters (causal) --------
            if col == "heart_rate":
                y = _jump_filter_causal(y, cfg.hr_max_jump)
            elif col == "derived_speed":
                y = _jump_filter_causal(y, cfg.speed_max_jump)
            elif col == "altitude":
                y = _jump_filter_causal(y, cfg.altitude_max_jump)

            # -------- causal imputation with recovery window --------
            if cfg.impute:
                y = _forward_fill_with_max_gap(y, cfg.max_gap)

            # If still many NaNs, apply fallback strategy
            if cfg.fallback_to_global_median_if_needed:
                if np.all(~np.isfinite(y)):
                    fill_val = _workout_fallback(y, col)
                    y[:] = fill_val
                else:
                    # fill leading NaNs (watch startup) with first available or workout/global median
                    if not np.isfinite(y[0]):
                        fill_val = _workout_fallback(y, col)
                        y = _fill_leading_nans(y, fill_val)

            # -------- clip using fitted bounds (doesn't use future) --------
            if cfg.clip:
                lo, hi = self.clip_bounds_[col]
                y = np.clip(y, lo, hi)

            # -------- final causal smoothing (EMA) --------
            if cfg.smooth and col in (cfg.smooth_cols or []):
                y = _ema_causal(y, cfg.ema_alpha)

            # ensure no NaNs remain (models usually require dense sequences)
            if np.any(~np.isfinite(y)):
                fill_val = _workout_fallback(y, col)
                y[~np.isfinite(y)] = fill_val

            return y.tolist()

        def clean_int_seq(x) -> list:
            y = _to_int_array(x)
            return y.tolist()

        for col in cfg.seq_float_cols:
            df_out[col] = df_out[col].apply(lambda v, c=col: clean_float_seq(v, c))
        for col in cfg.seq_int_cols:
            df_out[col] = df_out[col].apply(clean_int_seq)

        # Repair time_elapsed if present but invalid
        if "time_elapsed" in df_out.columns and "timestamp" in df_out.columns:

            def needs_fix(row) -> bool:
                return (
                    len(_as_list(row["time_elapsed"])) != len(_as_list(row["timestamp"]))
                    or len(_as_list(row["time_elapsed"])) == 0
                )

            def recompute(row) -> list:
                ts = _to_int_array(row["timestamp"])
                if ts.size == 0:
                    return []
                return (ts - ts[0]).astype(np.int64).tolist()

            mask = df_out.apply(needs_fix, axis=1)
            if mask.any():
                df_out.loc[mask, "time_elapsed"] = df_out.loc[mask].apply(recompute, axis=1)

        return df_out


# ---------------------------
# File operations
# ---------------------------
def clean_file(
    in_path: Path,
    out_path: Path,
    fit_on_path: Optional[Path],
    cfg: AdvancedCleaningConfig,
) -> None:
    df = read_parquet_pylist(in_path)

    cleaner = AdvancedSequenceCleaner(cfg)
    if fit_on_path is not None:
        df_fit = read_parquet_pylist(fit_on_path)
        cleaner.fit(df_fit)
    else:
        cleaner.fit(df)

    df_clean = cleaner.transform(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(out_path, index=False)
    print(f"Cleaned {in_path} -> {out_path} (rows {len(df)} -> {len(df_clean)})")


# ---------------------------
# CLI
# ---------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=None, help="Input parquet (single-file mode).")
    p.add_argument("--output", type=str, default=None, help="Output parquet (single-file mode).")
    p.add_argument("--fit-on", type=str, default=None, help="Optional parquet to fit bounds/medians on (baseline/train).")

    p.add_argument("--root", type=str, default=None, help="Root directory (directory mode).")
    p.add_argument("--pattern", type=str, default="**/*_test_raw.parquet", help="Glob pattern under --root.")

    p.add_argument("--mode", choices=["causal"], default="causal", help="Only causal mode in this advanced cleaner.")

    # dropping / validity
    p.add_argument("--min-seq-len", type=int, default=5)
    p.add_argument("--no-drop-invalid", action="store_true", help="Do NOT drop invalid workouts.")

    # clipping
    p.add_argument("--no-clip", action="store_true")
    p.add_argument("--clip-q-low", type=float, default=0.005)
    p.add_argument("--clip-q-high", type=float, default=0.995)

    # imputation / recovery
    p.add_argument("--no-impute", action="store_true")
    p.add_argument("--max-gap", type=int, default=15)

    # smoothing
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--ema-alpha", type=float, default=0.20)
    p.add_argument("--no-prefilter-ema", action="store_true")
    p.add_argument("--prefilter-alpha", type=float, default=0.12)

    # plausibility + jump thresholds (tuning knobs)
    p.add_argument("--hr-min", type=float, default=30.0)
    p.add_argument("--hr-max", type=float, default=230.0)
    p.add_argument("--hr-max-jump", type=float, default=50.0)

    p.add_argument("--speed-min", type=float, default=0.0)
    p.add_argument("--speed-max", type=float, default=12.0)
    p.add_argument("--speed-max-jump", type=float, default=6.0)

    p.add_argument("--altitude-max-jump", type=float, default=20.0)

    args = p.parse_args()

    cfg = AdvancedCleaningConfig(
        drop_invalid_length_rows=not args.no_drop_invalid,
        min_seq_len=args.min_seq_len,
        clip=not args.no_clip,
        clip_q_low=args.clip_q_low,
        clip_q_high=args.clip_q_high,
        impute=not args.no_impute,
        max_gap=args.max_gap,
        smooth=not args.no_smooth,
        ema_alpha=args.ema_alpha,
        prefilter_ema=not args.no_prefilter_ema,
        prefilter_alpha=args.prefilter_alpha,
        hr_min=args.hr_min,
        hr_max=args.hr_max,
        hr_max_jump=args.hr_max_jump,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        speed_max_jump=args.speed_max_jump,
        altitude_max_jump=args.altitude_max_jump,
    )

    fit_on_path = Path(args.fit_on) if args.fit_on else None

    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_cleaned.parquet")
        clean_file(in_path, out_path, fit_on_path, cfg)
        return

    if args.root:
        root = Path(args.root)
        if not root.exists():
            raise FileNotFoundError(root)

        for in_path in sorted(root.glob(args.pattern)):
            out_path = in_path.with_name(f"{in_path.stem}_cleaned.parquet")
            clean_file(in_path, out_path, fit_on_path, cfg)
        return

    raise SystemExit("Provide either --input <file> or --root <dir>.")


if __name__ == "__main__":
    main()
