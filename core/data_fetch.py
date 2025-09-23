"""Data fetching utilities built on top of yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Company:
    """Simple container for a company's ticker symbol."""

    ticker: str


def _ensure_companies(sequence: Iterable[dict | Company]) -> List[Company]:
    companies: List[Company] = []
    for item in sequence:
        if isinstance(item, Company):
            companies.append(item)
        elif isinstance(item, dict) and "ticker" in item:
            companies.append(Company(ticker=str(item["ticker"]).upper()))
        else:
            raise ValueError("Each company entry must provide a 'ticker'.")
    if not companies:
        raise ValueError("companies_list must contain at least one ticker entry.")
    return companies


def _flatten_fundamental_dict(
    fund_dict: dict[str, pd.DataFrame],
    *,
    label: str = "ticker",
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for ticker, frame in fund_dict.items():
        if frame is None or frame.empty:
            continue
        frames.append(
            frame.T.assign(**{label: ticker}).reset_index().rename(columns={"index": "Date"})
        )
    if not frames:
        return pd.DataFrame(columns=["Date", label])
    return pd.concat(frames, ignore_index=True)


def get_companies_quotes(companies_list: Iterable[dict | Company], period: str = "5y") -> pd.DataFrame:
    """Fetch historical quotes for a collection of tickers."""

    companies = _ensure_companies(companies_list)

    frames: List[pd.DataFrame] = []
    for company in companies:
        ticker_data = yf.Ticker(company.ticker)
        frame = ticker_data.history(period=period)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["ticker"] = company.ticker
        frames.append(frame)

    if not frames:
        raise ValueError("No quotes retrieved; check tickers or internet access.")

    all_quotes = pd.concat(frames)
    all_quotes = all_quotes.reset_index()
    all_quotes["Date"] = pd.to_datetime(all_quotes["Date"]).dt.date
    return all_quotes


def get_fundamental_data(
    companies_list: Iterable[dict | Company],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download company info and quarterly financial statements."""

    companies = _ensure_companies(companies_list)

    company_info: List[pd.Series] = []
    quarterly_balance_sheet: dict[str, pd.DataFrame] = {}
    quarterly_cashflow: dict[str, pd.DataFrame] = {}
    quarterly_income: dict[str, pd.DataFrame] = {}

    for company in companies:
        ticker = company.ticker
        print(f"Data download: {ticker}")
        ticker_data = yf.Ticker(ticker)

        info = pd.Series(ticker_data.info)
        info["ticker"] = ticker
        company_info.append(info)

        quarterly_balance_sheet[ticker] = ticker_data.quarterly_balance_sheet
        quarterly_cashflow[ticker] = ticker_data.quarterly_cashflow
        quarterly_income[ticker] = ticker_data.quarterly_income_stmt

    info_df = pd.DataFrame(company_info).reset_index(drop=True)
    income_df = _flatten_fundamental_dict(quarterly_income)
    balance_df = _flatten_fundamental_dict(quarterly_balance_sheet)
    cash_df = _flatten_fundamental_dict(quarterly_cashflow)

    return info_df, income_df, balance_df, cash_df
