#!/usr/bin/env python3

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--timestamp", default="timestamp")
    parser.add_argument("--target", default="demand_mw")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    problems = []

    if args.timestamp not in df.columns:
        problems.append("timestamp column missing")

    if args.target not in df.columns:
        problems.append("target column missing")

    if args.timestamp in df.columns:
        ts = pd.to_datetime(df[args.timestamp], errors="coerce")

        if ts.isna().any():
            problems.append("invalid timestamps detected")

        if ts.duplicated().any():
            problems.append("duplicate timestamps detected")

    if args.target in df.columns:
        if df[args.target].isna().any():
            problems.append("missing target values detected")

    if problems:
        print("NOT READY")
        for item in problems:
            print(f"- {item}")
        raise SystemExit(1)

    print("READY")
    print(f"Rows checked: {len(df):,}")


if __name__ == "__main__":
    main()
