#!/usr/bin/env python3

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--timestamp", default="timestamp")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    ts = (
        pd.to_datetime(df[args.timestamp], errors="coerce")
        .dropna()
        .sort_values()
    )

    diff = ts.diff().dropna()

    expected = pd.Timedelta(minutes=30)
    bad = diff[diff != expected]

    print(f"Intervals checked: {len(diff):,}")
    print(f"Non-30-minute intervals: {len(bad):,}")

    if len(bad):
        print("\nUnexpected intervals:")
        print(bad.value_counts().head(10).to_string())
    else:
        print("PASS: continuous 30-minute grid.")


if __name__ == "__main__":
    main()
