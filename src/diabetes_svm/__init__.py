"""Diabetes risk classification with Support Vector Machines."""

from .config import FEATURES, TARGET, ZERO_AS_MISSING
from .data import load_dataset, split_features_target
from .pipeline import build_pipeline, tune_pipeline
from .predict import RiskReport, predict_risk

__all__ = [
    "FEATURES",
    "TARGET",
    "ZERO_AS_MISSING",
    "load_dataset",
    "split_features_target",
    "build_pipeline",
    "tune_pipeline",
    "predict_risk",
    "RiskReport",
]
__version__ = "1.0.0"
