"""Configuration values specific to the feature_scaling package."""

from __future__ import annotations

from typing import Mapping

FEATURE_SCALING_CASES: Mapping[str, dict] = {
    "case_2": {
        "features": [
            "OpenCloseReturn",
            "IntradayRange",
            "rsi_14",
            "Volume_vs_MA14",
            "ROA",
            "ROA_pct_change",
            "Free Cash Flow",
            "Free Cash Flow_pct_change",
            "Tangible Book Value",
            "Tangible Book Value_pct_change",
        ],
        "target": "target_Close+14_pct",
        "target_source": "target_pct_14",
    },
    "case_3": {
        "features": [
            "OpenCloseReturn",
            "IntradayRange",
            "rsi_14",
            "Volume_vs_MA14",
            "ROA",
            "ROA_pct_change",
            "Free Cash Flow",
            "Free Cash Flow_pct_change",
            "Tangible Book Value",
            "Tangible Book Value_pct_change",
        ],
        "technical_features": [
            "OpenCloseReturn",
            "IntradayRange",
            "rsi_14",
            "Volume_vs_MA14",
        ],
        "fundamental_features": [
            "ROA",
            "ROA_pct_change",
            "Free Cash Flow",
            "Free Cash Flow_pct_change",
            "Tangible Book Value",
            "Tangible Book Value_pct_change",
        ],
        "target": "target_Close+14_pct",
        "target_source": "target_pct_14",
    },
    "case_4": {
        "features": [
            "OpenCloseReturn",
            "IntradayRange",
            "rsi_14",
            "Volume_vs_MA14",
        ],
        "target": "target_Close+14_pct",
        "target_source": "target_pct_14",
    },
    "case_5": {
        "features": [
            "OpenCloseReturn",
            "IntradayRange",
            "rsi_14",
            "Volume_vs_MA14",
        ],
        "target": "target_Close+14_pct",
        "target_source": "target_pct_14",
    },
}

__all__ = ["FEATURE_SCALING_CASES"]
