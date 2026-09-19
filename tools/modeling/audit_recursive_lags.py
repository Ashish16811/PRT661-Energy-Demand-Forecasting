#!/usr/bin/env python3

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Audit recursive forecast lag provenance."
    )
    parser.add_argument("csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    expected = {
        "timestamp",
        "forecast",
        "lag_source"
    }

    missing = expected - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing provenance fields: {sorted(missing)}"
        )

    print("Recursive lag provenance")
    print(df["lag_source"].value_counts(dropna=False).to_string())

    missing_source = df["lag_source"].isna().sum()

    print(f"\nRows without lag provenance: {missing_source:,}")

    if missing_source:
        raise SystemExit("FAIL: forecast rows lack lag provenance.")

    print("PASS: all forecast rows include lag provenance.")


if __name__ == "__main__":
    main()
