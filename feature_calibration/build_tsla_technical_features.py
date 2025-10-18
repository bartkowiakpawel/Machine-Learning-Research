"""Generate a classic technical indicator feature set for TSLA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("feature_calibration/data/tsla_yf_dataset.csv")
DEFAULT_OUTPUT = Path("feature_calibration/data/tsla_technical_features.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a dataset with classic technical indicators for TSLA based on the "
            "Yahoo Finance OHLCV feed."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the base TSLA dataset with OHLCV columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the enriched dataset with indicators.",
    )
    return parser.parse_args()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Simple moving averages
    for window in (10, 20, 50, 100, 200):
        data[f"sma_{window}"] = data["Close"].rolling(window).mean()

    # Exponential moving averages
    for span in (12, 26):
        data[f"ema_{span}"] = ema(data["Close"], span)

    # MACD and signal line
    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = ema(data["macd"], 9)
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    # Relative Strength Index (RSI 14)
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-day SMA, 2 std)
    sma_20 = data["Close"].rolling(20).mean()
    std_20 = data["Close"].rolling(20).std()
    data["bollinger_mid"] = sma_20
    data["bollinger_upper"] = sma_20 + 2 * std_20
    data["bollinger_lower"] = sma_20 - 2 * std_20
    data["bollinger_bandwidth"] = (data["bollinger_upper"] - data["bollinger_lower"]) / data["bollinger_mid"]

    # Stochastic Oscillator (14)
    lowest_low = data["Low"].rolling(window=14).min()
    highest_high = data["High"].rolling(window=14).max()
    data["stoch_k"] = 100 * (data["Close"] - lowest_low) / (highest_high - lowest_low)
    data["stoch_d"] = data["stoch_k"].rolling(window=3).mean()

    # Average True Range (ATR 14)
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["atr_14"] = true_range.rolling(window=14).mean()
    data["atr_ratio"] = data["atr_14"] / data["Close"]

    # Rate of Change (ROC) for multiple horizons
    for window in (5, 10, 20):
        data[f"roc_{window}"] = data["Close"].pct_change(periods=window)

    # Daily log return
    data["log_return_1d"] = np.log(data["Close"] / data["Close"].shift())

    return data


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    df = pd.read_csv(args.input, parse_dates=["Date"])
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    enriched = compute_indicators(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    print(f"Saved technical indicator dataset to {args.output}")


if __name__ == "__main__":
    main()
