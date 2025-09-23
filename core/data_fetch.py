"""Simplified Yahoo Finance fetch utilities."""

import pandas as pd
import yfinance as yf


def _collect_tickers(items):
    tickers = []
    for item in items:
        if isinstance(item, str):
            tickers.append(item.strip().upper())
        elif isinstance(item, dict):
            tickers.append(str(item.get("ticker", "")).strip().upper())
        else:
            tickers.append(str(getattr(item, "ticker", "")).strip().upper())
    return [t for t in tickers if t]

def get_companies_quotes(companies_list, period: str = "5y") -> pd.DataFrame:
    """Return historical quotes with a few handy indicators."""

    tickers = _collect_tickers(companies_list)
    if not tickers:
        raise ValueError("No tickers provided.")

    frames = []
    for ticker in tickers:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            continue
        data = data.copy()
        data["ticker"] = ticker

        frames.append(data)

    if not frames:
        raise ValueError("No quotes retrieved from Yahoo Finance.")

    out = pd.concat(frames).reset_index()
    out["Date"] = pd.to_datetime(out["Date"]).dt.date
    return out

def get_fundamental_data(companies_list):
    """Return company info and quarterly statements for the tickers."""

    tickers = _collect_tickers(companies_list)
    if not tickers:
        raise ValueError("No tickers provided.")

    info_rows = []
    income_frames = []
    balance_frames = []
    cash_frames = []

    for ticker in tickers:
        ticker_data = yf.Ticker(ticker)

        info = pd.Series(ticker_data.info)
        info["ticker"] = ticker
        info_rows.append(info)

        income = ticker_data.quarterly_income_stmt
        if income is not None and not income.empty:
            income = income.T.reset_index().rename(columns={"index": "Date"})
            income["ticker"] = ticker
            income_frames.append(income)

        balance = ticker_data.quarterly_balance_sheet
        if balance is not None and not balance.empty:
            balance = balance.T.reset_index().rename(columns={"index": "Date"})
            balance["ticker"] = ticker
            balance_frames.append(balance)

        cash = ticker_data.quarterly_cashflow
        if cash is not None and not cash.empty:
            cash = cash.T.reset_index().rename(columns={"index": "Date"})
            cash["ticker"] = ticker
            cash_frames.append(cash)

    info_df = pd.DataFrame(info_rows).reset_index(drop=True) if info_rows else pd.DataFrame()
    income_df = pd.concat(income_frames, ignore_index=True) if income_frames else pd.DataFrame(columns=["Date", "ticker"])
    balance_df = pd.concat(balance_frames, ignore_index=True) if balance_frames else pd.DataFrame(columns=["Date", "ticker"])
    cash_df = pd.concat(cash_frames, ignore_index=True) if cash_frames else pd.DataFrame(columns=["Date", "ticker"])

    return info_df, income_df, balance_df, cash_df
