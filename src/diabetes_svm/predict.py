"""Turn a fitted model into a readable risk report for one person."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .config import FEATURES, MODEL_PATH


@dataclass(frozen=True)
class RiskReport:
    label: str
    probability: float
    band: str

    def __str__(self) -> str:
        return f"{self.label} (estimated probability {self.probability:.0%}, {self.band} risk band)"


def _band(p: float) -> str:
    if p < 0.30:
        return "low"
    if p < 0.60:
        return "moderate"
    return "high"


def predict_risk(values: dict[str, float], model=None, model_path: str | Path = MODEL_PATH) -> RiskReport:
    """Predict diabetes risk from a mapping of feature name -> value.

    Any feature left out (or set to ``None``) is treated as unmeasured and is
    imputed by the pipeline.
    """
    unknown = set(values) - set(FEATURES)
    if unknown:
        raise KeyError(f"Unknown feature(s): {sorted(unknown)}")
    if model is None:
        model = joblib.load(model_path)
    row = pd.DataFrame([{name: values.get(name) for name in FEATURES}], dtype=float)
    probability = float(model.predict_proba(row)[0, 1])
    label = "Positive screening result" if probability >= 0.5 else "Negative screening result"
    return RiskReport(label=label, probability=probability, band=_band(probability))
