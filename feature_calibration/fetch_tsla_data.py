"""Utility script to fetch TSLA data directly from Yahoo Finance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_tsla(
    *,
    period: str = "10y",
    interval: str = "1d",
    output_path: Path | str | None = None,
) -> Path:
    """Download TSLA OHLCV data from Yahoo Finance and persist it locally."""

    ticker = yf.Ticker("TSLA")
    hist = ticker.history(period=period, interval=interval, auto_adjust=False, actions=True)
    if hist.empty:
        raise ValueError("Failed to retrieve data for TSLA from Yahoo Finance.")

    hist = hist.reset_index()
    hist = hist.rename(
        columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
            "Dividends": "Dividends",
            "Stock Splits": "StockSplits",
        }
    )
    hist["ticker"] = "TSLA"

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "data" / "tsla_yf_dataset.csv"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(output_path, index=False)

    print(f"Downloaded TSLA data to: {output_path}")
    return output_path


def main() -> None:
    fetch_tsla()


if __name__ == "__main__":
    main()
