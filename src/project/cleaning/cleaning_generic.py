#!/usr/bin/env python3
"""
Generic, extensible cleaner for FitRec-style Endomondo "raw" parquet files where sequential
signals are stored as list/array columns per workout.

This module focuses on cleaning `test_raw.parquet`

Default cleaning steps:
1) Validate sequence lengths (all sequence columns match timestamp length)
   - policy: drop invalid rows (configurable)
2) Coerce sequence values to numeric, turning bad values/inf into NaN
3) Impute NaNs per-sequence via linear interpolation + edge fill
4) Optional clipping using learned bounds (fit on provided training dataframe OR on current df)
5) Optional light smoothing (rolling median) for selected signals

Usage (single file):
  python cleaning_generic.py --input test_raw.parquet --output test_raw_cleaned.parquet

Usage (directory with walking/running/biking subdirs):
  python cleaning_generic.py --root path/to/artifacts --pattern "**/test_raw.parquet"
- It writes parquet with list columns as Python lists (object dtype), which is broadly compatible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SEQ_COLS_FLOAT_DEFAULT = [
    "derived_speed",
    "heart_rate",
    "longitude",
    "latitude",
    "derived_distance",
    "altitude",
]
SEQ_COLS_INT_DEFAULT = ["timestamp", "time_elapsed"]


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


def _interp_1d(y: np.ndarray) -> np.ndarray:
    """Linear interpolate NaNs; then fill edges. Leaves all-NaN arrays unchanged."""
    if y.size == 0:
        return y
    idx = np.arange(y.size)
    mask = np.isfinite(y)
    if mask.all():
        return y
    if mask.sum() == 0:
        return y
    y2 = y.copy()
    y2[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
    first = int(np.argmax(np.isfinite(y2)))
    last = int(y2.size - 1 - np.argmax(np.isfinite(y2[::-1])))
    y2[:first] = y2[first]
    y2[last + 1 :] = y2[last]
    return y2


def _rolling_median(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size == 0:
        return y
    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y)
    for i in range(y.size):
        out[i] = np.nanmedian(padded[i : i + window])
    return out

def _read_parquet_pylist(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    table = pq.read_table(path, use_pandas_metadata=False)

    # Convert each column to plain Python objects (lists, ints, floats, strings)
    data = {name: table[name].to_pylist() for name in table.column_names}

    return pd.DataFrame(data)



@dataclass
class CleaningConfig:
    seq_float_cols: List[str] = field(default_factory=lambda: SEQ_COLS_FLOAT_DEFAULT.copy())
    seq_int_cols: List[str] = field(default_factory=lambda: SEQ_COLS_INT_DEFAULT.copy())
    drop_invalid_length_rows: bool = True
    min_seq_len: int = 5
    impute: bool = True
    clip: bool = True
    clip_q_low: float = 0.005
    clip_q_high: float = 0.995
    smooth: bool = True
    smooth_window: int = 3
    smooth_cols: Optional[List[str]] = None  # defaults to hr + speed + altitude if None


class GenericSequenceCleaner:
    """Simple fit/transform cleaner for workout-level list-column parquets."""

    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.clip_bounds_: Optional[Dict[str, Tuple[float, float]]] = None
        self.fallback_median_: Optional[Dict[str, float]] = None

        if self.config.smooth_cols is None:
            self.config.smooth_cols = [
                c for c in ["heart_rate", "derived_speed", "altitude"] if c in self.config.seq_float_cols
            ]

    def _validate_schema(self, df: pd.DataFrame) -> None:
        required = ["id"] + self.config.seq_int_cols + self.config.seq_float_cols
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def fit(self, df_train: pd.DataFrame) -> "GenericSequenceCleaner":
        self._validate_schema(df_train)
        cfg = self.config

        bounds: Dict[str, Tuple[float, float]] = {}
        medians: Dict[str, float] = {}

        for col in cfg.seq_float_cols:
            all_vals = []
            for x in df_train[col].tolist():
                arr = _to_float_array(x)
                if arr.size:
                    all_vals.append(arr)
            if not all_vals:
                bounds[col] = (-np.inf, np.inf)
                medians[col] = np.nan
                continue

            v = np.concatenate(all_vals)
            v = v[np.isfinite(v)]
            if v.size == 0:
                bounds[col] = (-np.inf, np.inf)
                medians[col] = np.nan
                continue

            lo = float(np.quantile(v, cfg.clip_q_low))
            hi = float(np.quantile(v, cfg.clip_q_high))
            bounds[col] = (lo, hi)
            medians[col] = float(np.median(v))

        self.clip_bounds_ = bounds
        self.fallback_median_ = medians
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_schema(df)
        if self.clip_bounds_ is None or self.fallback_median_ is None:
            raise RuntimeError("Cleaner not fitted. Call fit() first.")

        cfg = self.config
        df_out = df.copy()

        # Force list columns to object dtype so we can safely write Python lists
        for col in cfg.seq_int_cols + cfg.seq_float_cols:
            df_out[col] = df_out[col].astype(object)

        # Validate lengths
        ts_lens = df_out["timestamp"].apply(lambda x: len(_as_list(x)))
        valid = ts_lens >= cfg.min_seq_len
        for col in cfg.seq_float_cols + cfg.seq_int_cols:
            valid &= df_out[col].apply(lambda x: len(_as_list(x))).eq(ts_lens)

        if cfg.drop_invalid_length_rows:
            df_out = df_out.loc[valid].reset_index(drop=True)

        def clean_float_seq(x, col: str) -> list:
            y = _to_float_array(x)
            if y.size == 0:
                return []
            y[~np.isfinite(y)] = np.nan
            if cfg.impute:
                y = _interp_1d(y)

            if np.all(~np.isfinite(y)):
                med = self.fallback_median_.get(col, np.nan)
                y[:] = med if np.isfinite(med) else 0.0

            if cfg.clip:
                lo, hi = self.clip_bounds_[col]
                y = np.clip(y, lo, hi)

            if cfg.smooth and col in (cfg.smooth_cols or []):
                y = _rolling_median(y, cfg.smooth_window)

            return y.tolist()

        def clean_int_seq(x) -> list:
            y = _to_int_array(x)
            return y.tolist()

        for col in cfg.seq_float_cols:
            df_out[col] = df_out[col].apply(lambda x, c=col: clean_float_seq(x, c))
        for col in cfg.seq_int_cols:
            df_out[col] = df_out[col].apply(clean_int_seq)

        # Recompute time_elapsed if empty or wrong length (optional repair)
        if "time_elapsed" in df_out.columns and "timestamp" in df_out.columns:
            def needs_fix(row) -> bool:
                return len(_as_list(row["time_elapsed"])) != len(_as_list(row["timestamp"])) or len(_as_list(row["time_elapsed"])) == 0

            def recompute(row) -> list:
                ts = _to_int_array(row["timestamp"])
                if ts.size == 0:
                    return []
                return (ts - ts[0]).astype(np.int64).tolist()

            mask = df_out.apply(needs_fix, axis=1)
            if mask.any():
                df_out.loc[mask, "time_elapsed"] = df_out.loc[mask].apply(recompute, axis=1)

        return df_out


def clean_file(in_path: Path, out_path: Path, fit_on_path: Optional[Path] = None) -> None:
    df = _read_parquet_pylist(in_path)

    cleaner = GenericSequenceCleaner()

    if fit_on_path is not None:
        df_fit = _read_parquet_pylist(fit_on_path)
        cleaner.fit(df_fit)
    else:
        cleaner.fit(df)

    df_clean = cleaner.transform(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(out_path, index=False)
    print(f"Cleaned {in_path} -> {out_path} (rows {len(df)} -> {len(df_clean)})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=None, help="Input parquet file (single-file mode).")
    p.add_argument("--output", type=str, default=None, help="Output parquet file (single-file mode).")
    p.add_argument("--fit-on", type=str, default=None, help="Optional parquet to fit bounds/medians on (e.g., train).")
    p.add_argument("--root", type=str, default=None, help="Root directory to search (directory mode).")
    p.add_argument("--pattern", type=str, default="**/test_raw.parquet", help="Glob pattern under --root.")
    args = p.parse_args()

    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_cleaned.parquet")
        fit_on_path = Path(args.fit_on) if args.fit_on else None
        clean_file(in_path, out_path, fit_on_path=fit_on_path)
        return

    if args.root:
        root = Path(args.root)
        if not root.exists():
            raise FileNotFoundError(root)
        fit_on_path = Path(args.fit_on) if args.fit_on else None
        for in_path in sorted(root.glob(args.pattern)):
            out_path = in_path.with_name(f"{in_path.stem}_cleaned.parquet")
            clean_file(in_path, out_path, fit_on_path=fit_on_path)
        return

    raise SystemExit("Provide either --input <file> or --root <dir>.")


if __name__ == "__main__":
    main()
