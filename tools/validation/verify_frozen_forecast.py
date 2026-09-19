#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Verify frozen forecasts against held-out actuals."
    )
    parser.add_argument("csv")
    parser.add_argument("--actual", default="actual")
    parser.add_argument("--forecast", default="forecast")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    data = df[[args.actual, args.forecast]].dropna()

    actual = data[args.actual].astype(float)
    forecast = data[args.forecast].astype(float)

    error = forecast - actual

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    bias = np.mean(error)

    denominator = np.abs(actual) + np.abs(forecast)
    valid = denominator > 0

    smape = (
        np.mean(
            2 * np.abs(error[valid]) / denominator[valid]
        ) * 100
    )

    print(f"Matched intervals: {len(data):,}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"sMAPE: {smape:.2f}%")
    print(f"Bias: {bias:+.2f}")


if __name__ == "__main__":
    main()
