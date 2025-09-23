"""Command-line entry point for running project demos."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from feature_scaling.california_housing_demo import run_demo as run_feature_scaling_california_demo

MenuOption = Tuple[str, Callable[[], None] | None]


def _prompt_menu(title: str, options: Dict[str, MenuOption]) -> str:
    """Display a menu and return the selected key."""

    print(f"\n{title}")
    for key, (label, _) in sorted(options.items()):
        print(f"{key}. {label}")

    return input("Choose an option: ").strip()


def _feature_scaling_menu() -> None:
    """Handle feature scaling sub-menu."""

    options: Dict[str, MenuOption] = {
        "1": ("California Housing - Yeo-Johnson demo", run_feature_scaling_california_demo),
        "0": ("Return to main menu", None),
    }

    while True:
        choice = _prompt_menu("Feature Scaling - available demos:", options)

        if choice == "0":
            return

        action = options.get(choice, ("", None))[1]
        if action is None:
            print("Invalid choice, please try again.")
            continue

        action()


def main() -> None:
    """Main CLI entry point."""

    options: Dict[str, MenuOption] = {
        "1": ("Feature Scaling", _feature_scaling_menu),
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
