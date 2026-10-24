"""Figure factory. Every function returns a Matplotlib ``Figure``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .config import FEATURES, RANDOM_STATE, TARGET

PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
LABELS = {0: "Non-diabetic", 1: "Diabetic"}
sns.set_theme(style="whitegrid", context="notebook")


def _style(fig):
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def class_balance_and_missing(frame: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = frame[TARGET].value_counts().sort_index()
    bars = axes[0].bar([LABELS[i] for i in counts.index], counts.values, color=[PALETTE[i] for i in counts.index])
    for bar, value in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 8, f"{value} ({value / len(frame):.0%})", ha="center")
    axes[0].set_title("Class balance")
    axes[0].set_ylabel("Patients")
    axes[0].set_ylim(0, counts.max() * 1.15)

    missing = frame[FEATURES].isna().mean().mul(100).sort_values()
    axes[1].barh(missing.index, missing.values, color="#457b9d")
    axes[1].set_title("Missing values after treating impossible zeros as NaN")
    axes[1].set_xlabel("% of rows")
    return _style(fig)


def feature_distributions(frame: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, feature in zip(axes.ravel(), FEATURES):
        for cls in (0, 1):
            sns.kdeplot(frame.loc[frame[TARGET] == cls, feature].dropna(), ax=ax, fill=True,
                        alpha=0.35, color=PALETTE[cls], label=LABELS[cls], warn_singular=False)
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Feature distributions by outcome", y=1.01, fontsize=14)
    return _style(fig)


def correlation_heatmap(frame: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    corr = frame[FEATURES + [TARGET]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_title("Feature correlation matrix")
    return _style(fig)


def kernel_comparison(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    table.plot.bar(ax=ax, color=["#264653", "#e9c46a"], rot=0)
    ax.set_ylim(0.6, 0.9)
    ax.set_title("Kernel comparison (5-fold cross-validation)")
    ax.set_ylabel("Score")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)
    return _style(fig)


def evaluation_panel(model, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=[LABELS[0], LABELS[1]], cmap="Blues", colorbar=False, ax=axes[0]
    )
    axes[0].set_title("Confusion matrix (held-out test set)")
    axes[0].grid(False)
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1], curve_kwargs={"color": "#e76f51", "lw": 2})
    axes[1].plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    axes[1].set_title("ROC curve")
    axes[1].legend(loc="lower right")
    return _style(fig)


def permutation_importance_plot(model, X_test, y_test, repeats: int = 30):
    result = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=repeats,
                                    random_state=RANDOM_STATE)
    order = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(np.array(FEATURES)[order], result.importances_mean[order], xerr=result.importances_std[order],
            color="#2a9d8f", capsize=3)
    ax.set_title("Permutation importance (drop in ROC-AUC when a feature is shuffled)")
    ax.set_xlabel("Mean AUC decrease")
    return _style(fig)


def decision_boundary(X: pd.DataFrame, y: pd.Series, build, features=("Glucose", "BMI")):
    """Fit a 2-feature RBF model purely to visualise how the margin looks."""
    sub = X[list(features)]
    model = build(kernel="rbf").fit(sub, y)
    pad = 0.08
    lo = sub.min() - (sub.max() - sub.min()) * pad
    hi = sub.max() + (sub.max() - sub.min()) * pad
    xx, yy = np.meshgrid(np.linspace(lo.iloc[0], hi.iloc[0], 300), np.linspace(lo.iloc[1], hi.iloc[1], 300))
    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=list(features))
    zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    contour = ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r", alpha=0.75)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.8)
    ax.scatter(sub.iloc[:, 0], sub.iloc[:, 1], c=y.map(PALETTE), edgecolor="white", s=22, linewidth=0.4)
    fig.colorbar(contour, ax=ax, label="Predicted probability of diabetes")
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title(f"Decision surface using only {features[0]} and {features[1]}")
    return _style(fig)


def workflow_diagram():
    """Draw the training/inference workflow as an image (README fallback for Mermaid)."""
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    stages = [
        ("Raw CSV", "768 rows\n8 features", "#e9ecef"),
        ("Clean", "zeros -> NaN\nvalidate schema", "#cde7f0"),
        ("Split", "80 / 20\nstratified", "#cde7f0"),
        ("Impute", "median\n(train only)", "#fde8c8"),
        ("Scale", "StandardScaler\n(train only)", "#fde8c8"),
        ("SVM", "grid-search\nkernel, C, gamma", "#f9c6bd"),
        ("Report", "metrics, plots\nrisk probability", "#d4edda"),
    ]
    width, gap, y0 = 1.5, 0.5, 1.0
    for i, (title, body, color) in enumerate(stages):
        x = 0.25 + i * (width + gap)
        ax.add_patch(FancyBboxPatch((x, y0), width, 1.6, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc=color, ec="#343a40", lw=1.2))
        ax.text(x + width / 2, y0 + 1.15, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + width / 2, y0 + 0.55, body, ha="center", va="center", fontsize=8.5, color="#343a40")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.03, y0 + 0.8), (x + width + gap - 0.03, y0 + 0.8),
                                         arrowstyle="-|>", mutation_scale=14, color="#343a40"))
    ax.add_patch(FancyBboxPatch((0.25 + 3 * (width + gap) - 0.18, 0.55), 3 * width + 2 * gap + 0.36, 2.5,
                                boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec="#6c757d",
                                lw=1.2, ls="--"))
    ax.text(0.25 + 3 * (width + gap) + (3 * width + 2 * gap) / 2, 0.25, "single scikit-learn Pipeline (no data leakage)",
            ha="center", fontsize=9.5, style="italic", color="#495057")
    return fig


def margin_concept_diagram():
    """Minimal 2-D illustration of hyperplane, margin and support vectors."""
    rng = np.random.default_rng(3)
    a = rng.normal([-2, -1.5], 0.7, (25, 2))
    b = rng.normal([2, 1.5], 0.7, (25, 2))
    from sklearn.svm import SVC

    X = np.vstack([a, b])
    y = np.array([0] * 25 + [1] * 25)
    clf = SVC(kernel="linear", C=1).fit(X, y)
    w, b0 = clf.coef_[0], clf.intercept_[0]
    xs = np.linspace(-4.5, 4.5, 50)
    line = lambda c: -(w[0] * xs + b0 - c) / w[1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(*a.T, c=PALETTE[0], s=40, label="Class 0")
    ax.scatter(*b.T, c=PALETTE[1], s=40, label="Class 1")
    ax.scatter(*clf.support_vectors_.T, s=170, facecolors="none", edgecolors="black", lw=1.5, label="Support vectors")
    ax.plot(xs, line(0), "k-", lw=2, label="Hyperplane")
    ax.plot(xs, line(1), "k--", lw=1)
    ax.plot(xs, line(-1), "k--", lw=1)
    ax.fill_between(xs, line(-1), line(1), color="grey", alpha=0.12, label="Margin")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_title("Linear SVM: hyperplane, margin and support vectors")
    ax.legend(loc="upper left", fontsize=8)
    return _style(fig)
