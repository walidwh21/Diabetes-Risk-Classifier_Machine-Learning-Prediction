"""Evaluation helpers that go beyond a single accuracy number."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import CV_FOLDS, RANDOM_STATE


def score_model(model, X, y) -> dict[str, float]:
    """Compute the standard binary-classification metrics for a fitted model."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall (sensitivity)": recall_score(y, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def cross_validated_scores(model, X, y) -> pd.DataFrame:
    """Mean and standard deviation of key metrics over stratified folds."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    raw = cross_validate(model, X, y, cv=cv, scoring=scoring)
    rows = {m: (raw[f"test_{m}"].mean(), raw[f"test_{m}"].std()) for m in scoring}
    return pd.DataFrame(rows, index=["mean", "std"]).T.round(3)


def compare_kernels(build, X, y, kernels=("linear", "poly", "rbf", "sigmoid")) -> pd.DataFrame:
    """Cross-validate one pipeline per kernel using default hyper-parameters."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    records = []
    for kernel in kernels:
        res = cross_validate(build(kernel=kernel), X, y, cv=cv, scoring=["accuracy", "roc_auc"])
        records.append(
            {
                "kernel": kernel,
                "accuracy": res["test_accuracy"].mean(),
                "roc_auc": res["test_roc_auc"].mean(),
            }
        )
    return pd.DataFrame(records).set_index("kernel").round(3)
