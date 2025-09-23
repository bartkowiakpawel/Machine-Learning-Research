"""Project-wide configuration constants."""

from pathlib import Path

DATA_DIR = Path("data")
YAHOO_DATA_DIR = DATA_DIR / "yahoo"

DEFAULT_STOCKS = [
    {"market": "NASDAQ", "ticker": "TSLA", "name": "Tesla, Inc."},
    {"market": "NASDAQ", "ticker": "AAPL", "name": "Apple Inc."},
    {"market": "NASDAQ", "ticker": "MSFT", "name": "Microsoft Corporation"},
    {"market": "NASDAQ", "ticker": "AMZN", "name": "Amazon.com, Inc."},
    {"market": "NASDAQ", "ticker": "GOOGL", "name": "Alphabet Inc. (Google)"},
    {"market": "NASDAQ", "ticker": "NVDA", "name": "NVIDIA Corporation"},
    {"market": "NYSE", "ticker": "JPM", "name": "JPMorgan Chase & Co."},
    {"market": "NYSE", "ticker": "JNJ", "name": "Johnson & Johnson"},
    {"market": "NYSE", "ticker": "XOM", "name": "Exxon Mobil Corporation"},
    {"market": "NASDAQ", "ticker": "META", "name": "Meta Platforms, Inc."},
    {"market": "NYSE", "ticker": "V", "name": "Visa Inc."},
    {"market": "NYSE", "ticker": "BRK-B", "name": "Berkshire Hathaway Inc. Class B"},
]

DEFAULT_TECH_WINDOWS = (14, 50)

