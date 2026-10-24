"""Loading and light cleaning of the diabetes table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PATH, FEATURES, TARGET, ZERO_AS_MISSING


def load_dataset(path: str | Path = DATA_PATH, zeros_to_nan: bool = True) -> pd.DataFrame:
    """Read the CSV and validate that the expected columns exist.

    Parameters
    ----------
    zeros_to_nan:
        When True, impossible zeros (e.g. a blood pressure of 0) are replaced
        by ``NaN`` so the pipeline's imputer can handle them explicitly.
    """
    frame = pd.read_csv(path)
    missing_cols = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_cols)}")
    if zeros_to_nan:
        frame[ZERO_AS_MISSING] = frame[ZERO_AS_MISSING].replace(0, np.nan)
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with the feature columns in a fixed order."""
    return frame[FEATURES].copy(), frame[TARGET].copy()


def describe_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column."""
    counts = frame.isna().sum()
    return pd.DataFrame({"missing": counts, "percent": (counts / len(frame) * 100).round(1)})
