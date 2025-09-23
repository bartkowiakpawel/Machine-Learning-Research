"""Basic helpers for turning Yahoo quotes into ML-ready indicators."""

from pathlib import Path

import numpy as np
import pandas as pd
import ta


def load_quotes(quotes_path):
    """Read the quotes CSV and normalise key columns."""

    quotes_path = Path(quotes_path)
    data = pd.read_csv(quotes_path)
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])
    if "ticker" in data.columns:
        data["ticker"] = data["ticker"].astype(str).str.upper()
    return data


def _weighted_moving_average(values, window):
    weights = np.arange(1, window + 1)
    return values.rolling(window).apply(
        lambda arr: np.dot(arr, weights[-len(arr):]) / weights[-len(arr):].sum(), raw=True
    )


def compute_technical_indicators(quotes, windows):
    """Calculate a richer set of technical indicators per ticker."""

    clean_windows = [int(w) for w in windows if isinstance(w, int) and int(w) > 0]
    if not clean_windows:
        return quotes.head(0).copy()

    frames = []
    for ticker, raw_group in quotes.groupby("ticker"):
        group = raw_group.sort_values("Date").reset_index(drop=True).copy()

        group["OpenCloseReturn"] = (group["Close"] - group["Open"]) / group["Open"] * 100
        group["IntradayRange"] = (group["High"] - group["Low"]) / group["Low"] * 100
        group["RSI_14"] = ta.momentum.RSIIndicator(close=group["Close"], window=14).rsi()
        group["Volume_vs_MA14"] = group["Volume"] / group["Volume"].rolling(window=14).mean()

        for window in clean_windows:
            close = group["Close"]
            high = group["High"]
            low = group["Low"]
            volume = group["Volume"]
            value_ccy = volume * close

            group[f"sma_{window}"] = close.rolling(window=window).mean()
            group[f"ema_{window}"] = ta.trend.EMAIndicator(close=close, window=window).ema_indicator()
            group[f"wma_{window}"] = _weighted_moving_average(close, window)
            group[f"stddev_{window}"] = close.rolling(window=window).std()
            group[f"volatility_{window}"] = close.rolling(window=window).std()
            group[f"rsi_{window}"] = ta.momentum.RSIIndicator(close=close, window=window).rsi()
            group[f"avg_volume_{window}"] = volume.rolling(window=window).mean()
            group[f"avg_volume_ccy_{window}"] = value_ccy.rolling(window=window).mean()
            group[f"slope_{window}"] = close.rolling(window=window).apply(
                lambda arr: pd.Series(arr).pct_change().mean(), raw=False
            )

            stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=window)
            group[f"stoch_k_{window}"] = stoch.stoch()
            group[f"stoch_d_{window}"] = stoch.stoch_signal()
            cci = ta.trend.CCIIndicator(high=high, low=low, close=close, window=window)
            group[f"cci_{window}"] = cci.cci()
            atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=window)
            group[f"atr_{window}"] = atr.average_true_range()
            adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=window)
            group[f"adx_{window}"] = adx.adx()
            group[f"dmi_plus_{window}"] = adx.adx_pos()
            group[f"dmi_minus_{window}"] = adx.adx_neg()

            future_close = close.shift(-window)
            group[f"target_{window}"] = future_close - close
            group[f"target_pct_{window}"] = (future_close - close) / close * 100

        macd = ta.trend.MACD(close=group["Close"])
        group["macd"] = macd.macd()
        group["macd_signal"] = macd.macd_signal()
        group["macd_diff"] = macd.macd_diff()

        bb = ta.volatility.BollingerBands(close=group["Close"], window=14)
        group["bb_high_14"] = bb.bollinger_hband()
        group["bb_low_14"] = bb.bollinger_lband()
        group["bb_mavg_14"] = bb.bollinger_mavg()
        group["bb_width_14"] = group["bb_high_14"] - group["bb_low_14"]

        frames.append(group)

    if not frames:
        return quotes.head(0).copy()

    result = pd.concat(frames, ignore_index=True)
    result = result.dropna().reset_index(drop=True)
    result = result.sort_values(["ticker", "Date"]).reset_index(drop=True)
    return result


def _load_statement(path, prefix):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)
    if data.empty:
        return data

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).reset_index(drop=True)
    data["ticker"] = data["ticker"].astype(str).str.upper()

    rename_map = {col: f"{prefix}_{col}" for col in data.columns if col not in ("Date", "ticker")}
    return data.rename(columns=rename_map)


def load_quarterly_fundamentals(data_dir):
    """Load and combine quarterly income, balance and cashflow statements."""

    data_dir = Path(data_dir)
    parts = [
        ("income", data_dir / "quarterly_income.csv"),
        ("balance", data_dir / "quarterly_balance.csv"),
        ("cash", data_dir / "quarterly_cashflows.csv"),
    ]

    frames = []
    for prefix, path in parts:
        df = _load_statement(path, prefix)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    fundamentals = frames[0]
    for df in frames[1:]:
        fundamentals = fundamentals.merge(df, on=["Date", "ticker"], how="outer")

    fundamentals = fundamentals.sort_values(["ticker", "Date"]).reset_index(drop=True)
    return fundamentals


def merge_with_fundamentals(technicals, fundamentals):
    """Attach nearest historical fundamentals to each technical observation."""

    if fundamentals.empty:
        return technicals.copy()

    tech = technicals.copy()
    fund = fundamentals.copy()

    tech["Date"] = pd.to_datetime(tech["Date"])
    fund["Date"] = pd.to_datetime(fund["Date"])
    tech["ticker"] = tech["ticker"].astype(str).str.upper()
    fund["ticker"] = fund["ticker"].astype(str).str.upper()

    tech = tech.dropna(subset=["Date"]).sort_values(["Date", "ticker"], kind="mergesort").reset_index(drop=True)
    fund = fund.dropna(subset=["Date"]).sort_values(["Date", "ticker"], kind="mergesort").reset_index(drop=True)

    merged = pd.merge_asof(
        tech,
        fund,
        on="Date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )

    technical_columns = list(technicals.columns)
    fundamental_columns = [col for col in merged.columns if col not in technical_columns]
    if fundamental_columns:
        merged = merged.dropna(subset=fundamental_columns, how="all")

    return merged.reset_index(drop=True)


def prepare_ml_dataset(quotes_path, windows=(14, 50), output_path=None):
    """Create a machine learning dataset with technical indicators and fundamentals."""

    quotes = load_quotes(quotes_path)
    technicals = compute_technical_indicators(quotes, windows)

    data_dir = Path(quotes_path).parent
    fundamentals = load_quarterly_fundamentals(data_dir)
    dataset = merge_with_fundamentals(technicals, fundamentals)

    if dataset.empty:
        raise ValueError("No rows available after merging technicals with fundamentals.")

    technicals_path = Path(quotes_path).with_name("technical_indicators.csv")
    technicals.to_csv(technicals_path, index=False)

    if output_path is None:
        output_path = Path(quotes_path).with_name("ml_dataset.csv")
    else:
        output_path = Path(output_path)

    dataset.to_csv(output_path, index=False)
    return output_path
