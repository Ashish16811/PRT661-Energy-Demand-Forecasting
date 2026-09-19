#!/usr/bin/env python3

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--timestamp", default="timestamp")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    ts = pd.to_datetime(df[args.timestamp], errors="coerce")
    ts = ts.dropna().sort_values()

    intervals = ts.diff().dropna().dt.total_seconds().div(60)

    print("WA source frequency summary")
    print(intervals.value_counts().sort_index().to_string())

    common = intervals.value_counts().head(5)

    print("\nMost common intervals:")
    print(common.to_string())


if __name__ == "__main__":
    main()
