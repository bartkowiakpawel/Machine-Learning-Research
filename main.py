"""Command-line entry point for running project demos."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Callable, Dict, Tuple

from config import DEFAULT_STOCKS, DEFAULT_TECH_WINDOWS, ML_INPUT_DIR, YAHOO_DATA_DIR
from feature_scaling import CASE_STUDIES, run_all_cases
from core.data_fetch import get_companies_quotes, get_fundamental_data
from core.data_preparation import prepare_ml_dataset

MenuOption = Tuple[str, Callable[[], None] | None]


def _prompt_menu(title: str, options: Dict[str, MenuOption]) -> str:
    """Display a menu and return the selected key."""

    print(f"\n{title}")
    for key, (label, _) in sorted(options.items()):
        print(f"{key}. {label}")

    return input("Choose an option: ").strip()

def _download_yahoo_data() -> None:
    """Download quotes and fundamentals for selected tickers and store them locally."""

    print("Default ticker universe:")
    for stock in DEFAULT_STOCKS:
        print(f" - {stock['ticker']}: {stock['name']}")

    selected = DEFAULT_STOCKS

    if not selected:
        print("No tickers selected; aborting download.")
        return

    period = input("Yahoo Finance period (default 5y): ").strip() or "5y"
    download_quotes = input("Download historical quotes? [Y/n]: ").strip().lower() != 'n'
    download_fundamentals = input("Download fundamentals? [Y/n]: ").strip().lower() != 'n'

    if not download_quotes and not download_fundamentals:
        print("Nothing selected to download.")
        return

    output_dir = YAHOO_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "tickers": [stock.get("ticker", "") for stock in selected],
        "period": period,
        "download_quotes": download_quotes,
        "download_fundamentals": download_fundamentals,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    quotes_path = None
    if download_quotes:
        try:
            quotes = get_companies_quotes(selected, period=period)
            quotes_path = output_dir / "quotes.csv"
            quotes.to_csv(quotes_path, index=False)
            print(f"Quotes saved to {quotes_path}")
        except Exception as exc:
            print(f"Failed to download quotes: {exc}")
        else:
            metadata["quotes_path"] = str(quotes_path)

    if download_fundamentals:
        try:
            info_df, income_df, balance_df, cash_df = get_fundamental_data(selected)
            info_path = output_dir / "company_info.csv"
            income_path = output_dir / "quarterly_income.csv"
            balance_path = output_dir / "quarterly_balance.csv"
            cash_path = output_dir / "quarterly_cashflows.csv"
            info_df.to_csv(info_path, index=False)
            income_df.to_csv(income_path, index=False)
            balance_df.to_csv(balance_path, index=False)
            cash_df.to_csv(cash_path, index=False)
            print("Fundamentals saved to:")
            print(f" - {info_path}")
            print(f" - {income_path}")
            print(f" - {balance_path}")
            print(f" - {cash_path}")
            metadata.update({
                "info_path": str(info_path),
                "income_path": str(income_path),
                "balance_path": str(balance_path),
                "cash_path": str(cash_path),
            })
        except Exception as exc:
            print(f"Failed to download fundamentals: {exc}")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata saved to {metadata_path}")


def _prepare_ml_data() -> None:
    """Build a machine-learning ready dataset from the current Yahoo snapshot."""

    quotes_path = YAHOO_DATA_DIR / "quotes.csv"
    if not quotes_path.exists():
        print("Quotes file not found. Run a download first.")
        return

    windows = list(DEFAULT_TECH_WINDOWS)

    ML_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        output_path = prepare_ml_dataset(quotes_path, windows=tuple(windows))
    except Exception as exc:
        print(f"Failed to prepare dataset: {exc}")
        return

    print()
    technicals_path = output_path.parent / "technical_indicators.csv"
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    print("Machine learning dataset prepared:")
    print(f" Source quotes: {quotes_path}")
    print(f" Technical indicators: {technicals_path}")
    print(f" ML dataset: {output_path}")
    print(f" Metadata summary: {metadata_path}")
    print(f" Output directory: {output_path.parent}")
    print(f" Windows used: {', '.join(str(w) for w in windows)}")


def _feature_scaling_menu() -> None:
    """Handle feature scaling sub-menu."""

    while True:
        print("\nFeature Scaling - available case studies:")
        print("1. Run all case studies")
        for idx, case in enumerate(CASE_STUDIES, start=2):
            print(f"{idx}. {case.case_id} - {case.title}")
        print("0. Return to main menu")

        choice = input("Choose an option: ").strip()

        if choice == "0":
            return
        if choice == "1":
            run_all_cases()
            continue

        try:
            numeric_choice = int(choice)
        except ValueError:
            print("Invalid choice, please try again.")
            continue

        case_index = numeric_choice - 2
        if 0 <= case_index < len(CASE_STUDIES):
            CASE_STUDIES[case_index].runner()
        else:
            print("Invalid choice, please try again.")


def main() -> None:
    """Main CLI entry point."""

    options: Dict[str, MenuOption] = {
        "1": ("Download Yahoo Finance data", _download_yahoo_data),
        "2": ("Prepare Yahoo Finance data for ML", _prepare_ml_data),
        "3": ("Feature Scaling", _feature_scaling_menu),
        "0": ("Exit", None),
    }

    while True:
        choice = _prompt_menu("Main menu:", options)

        if choice == "0":
            print("Goodbye!")
            return

        action = options.get(choice, ("", None))[1]
        if action is None:
            print("Invalid choice, please try again.")
            continue

        action()


if __name__ == "__main__":
    main()
