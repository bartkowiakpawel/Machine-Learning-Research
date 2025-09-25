"""Project-wide configuration constants."""

from pathlib import Path
from typing import Dict, Tuple

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ModuleNotFoundError:
    LGBMRegressor = None

DATA_DIR = Path("data")
YAHOO_DATA_DIR = DATA_DIR / "yahoo"
ML_INPUT_DIR = DATA_DIR / "ml_input"

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

# Optional dependencies: skip models whose libraries are not installed.
MODEL_SPECS = {
    "RandomForest": (RandomForestRegressor, {"n_estimators": 200, "random_state": 42}),
    "XGBoost": (XGBRegressor, {"n_estimators": 200, "random_state": 42, "verbosity": 0}),
    "GradientBoosting": (GradientBoostingRegressor, {"n_estimators": 200, "random_state": 42}),
    "Lasso": (Lasso, {"alpha": 0.01, "random_state": 42}),
    "Ridge": (Ridge, {"alpha": 1.0, "random_state": 42}),
    "LightGBM": (LGBMRegressor, {"n_estimators": 200, "random_state": 42, "verbose": -1}),
    "KNN": (KNeighborsRegressor, {"n_neighbors": 5, "weights": "distance"}),
    "LinearRegression": (LinearRegression, {}),
    "SVR": (SVR, {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}),
}

MODEL_DICT: Dict[str, object] = {}
_missing_models = []
for name, (estimator_cls, params) in MODEL_SPECS.items():
    if estimator_cls is None:
        _missing_models.append(name)
        continue
    MODEL_DICT[name] = estimator_cls(**params)

MISSING_MODELS: Tuple[str, ...] = tuple(_missing_models)
del _missing_models
