#!/usr/bin/env python
"""Train, tune, evaluate and persist the SVM pipeline; regenerate README figures.

Usage:  python scripts/train.py [--no-figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from diabetes_svm import build_pipeline, load_dataset, split_features_target, tune_pipeline  # noqa: E402
from diabetes_svm.config import ASSETS_DIR, MODEL_PATH, RANDOM_STATE, TEST_SIZE  # noqa: E402
from diabetes_svm.evaluate import compare_kernels, cross_validated_scores, score_model  # noqa: E402
from diabetes_svm import plots  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true", help="skip image generation")
    args = parser.parse_args()

    frame = load_dataset()
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"train={len(X_train)}  test={len(X_test)}")

    kernels = compare_kernels(build_pipeline, X_train, y_train)
    print("\nKernel comparison (CV on training split):\n", kernels)

    search = tune_pipeline(X_train, y_train)
    best = search.best_estimator_
    print("\nBest params:", search.best_params_, f"| CV ROC-AUC = {search.best_score_:.3f}")

    cv_table = cross_validated_scores(build_pipeline(**{k.split('__')[1]: v for k, v in search.best_params_.items()}),
                                      X_train, y_train)
    test_scores = score_model(best, X_test, y_test)
    print("\nHeld-out test metrics:")
    for name, value in test_scores.items():
        print(f"  {name:22s} {value:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    metrics_path = MODEL_PATH.with_name("metrics.json")
    metrics_path.write_text(json.dumps(
        {"best_params": {k: (v if isinstance(v, (int, float)) else str(v)) for k, v in search.best_params_.items()},
         "cv_roc_auc": round(search.best_score_, 4),
         "test": {k: round(v, 4) for k, v in test_scores.items()},
         "kernel_comparison": kernels.to_dict()}, indent=2))
    print(f"\nSaved model -> {MODEL_PATH.relative_to(ROOT)}")

    if args.no_figures:
        return
    ASSETS_DIR.mkdir(exist_ok=True)
    figures = {
        "01_class_balance_missing.png": plots.class_balance_and_missing(frame),
        "02_feature_distributions.png": plots.feature_distributions(frame),
        "03_correlation_heatmap.png": plots.correlation_heatmap(frame),
        "04_kernel_comparison.png": plots.kernel_comparison(kernels),
        "05_evaluation.png": plots.evaluation_panel(best, X_test, y_test),
        "06_permutation_importance.png": plots.permutation_importance_plot(best, X_test, y_test),
        "07_decision_surface.png": plots.decision_boundary(X, y, build_pipeline),
        "diagram_workflow.png": plots.workflow_diagram(),
        "diagram_svm_margin.png": plots.margin_concept_diagram(),
    }
    for name, fig in figures.items():
        fig.savefig(ASSETS_DIR / name, dpi=130, bbox_inches="tight")
        print("figure ->", f"assets/{name}")
    cv_table.to_csv(ASSETS_DIR / "cv_metrics.csv")


if __name__ == "__main__":
    main()
