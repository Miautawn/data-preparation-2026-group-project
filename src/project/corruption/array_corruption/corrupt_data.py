#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import pyarrow.parquet as pq

from ConstantNoiseCorruption import ConstantNoiseCorruption
from RandomNoiseCorruption import RandomNoiseCorruption


def read_parquet_pylist(path: Path) -> pd.DataFrame:
    # Robust: avoids pandas/pyarrow extension dtype issues for list columns
    table = pq.read_table(path, use_pandas_metadata=False)
    data = {name: table[name].to_pylist() for name in table.column_names}
    return pd.DataFrame(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input raw parquet (list-columns), e.g. biking_test_raw.parquet")
    ap.add_argument("--output", default=None, help="Output parquet; default adds _corrupted")
    ap.add_argument("--method", choices=["random", "constant"], default="random")
    ap.add_argument("--std-scale", type=float, default=2.0)
    ap.add_argument("--row-fraction", type=float, default=1.0)
    ap.add_argument("--segment-fraction", type=float, default=0.2)
    ap.add_argument("--num-segments", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--columns",
        nargs="+",
        default=["heart_rate", "derived_speed", "altitude"],
        help="Sequence columns to corrupt",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_corrupted.parquet")

    df = read_parquet_pylist(in_path)

    if args.method == "random":
        corrupter = RandomNoiseCorruption(
            columns=args.columns,
            row_fraction=args.row_fraction,
            segment_fraction=args.segment_fraction,
            num_segments=args.num_segments,
            std_scale=args.std_scale,
            seed=args.seed,
        )
    else:
        corrupter = ConstantNoiseCorruption(
            columns=args.columns,
            row_fraction=args.row_fraction,
            segment_fraction=args.segment_fraction,
            num_segments=args.num_segments,
            std_scale=args.std_scale,
            seed=args.seed,
        )

    df_corrupted = corrupter.transform(df)
    df_corrupted.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
