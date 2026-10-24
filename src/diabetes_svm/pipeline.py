"""Model construction: imputation -> scaling -> SVM, wrapped in one Pipeline.

Keeping every step inside a single ``Pipeline`` means that imputation medians
and scaling statistics are learned from the training folds only, which avoids
leaking information from the test data.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CV_FOLDS, RANDOM_STATE

PARAM_GRID = [
    {"svm__kernel": ["linear"], "svm__C": [0.01, 0.1, 1, 10]},
    {
        "svm__kernel": ["rbf"],
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.1],
    },
]


def build_pipeline(kernel: str = "linear", C: float = 1.0, **svm_kwargs) -> Pipeline:
    """Create an un-fitted preprocessing + SVM pipeline."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("svm", SVC(kernel=kernel, C=C, probability=True, random_state=RANDOM_STATE, **svm_kwargs)),
        ]
    )


def tune_pipeline(X, y, scoring: str = "roc_auc", n_jobs: int = -1) -> GridSearchCV:
    """Grid-search kernel and regularisation settings with stratified CV."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        build_pipeline(),
        param_grid=PARAM_GRID,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    return search.fit(X, y)
