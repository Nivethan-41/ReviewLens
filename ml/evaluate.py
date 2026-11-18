"""Evaluation module for ReviewLens fake review detection model.

Evaluates the saved model pipeline on the untouched held-out test set
exactly once per run, computing:
- Accuracy, precision, recall, F1 per class, and macro-F1
- ROC-AUC score
- Confusion matrix visualization (saved to models/reports/confusion_matrix.png)
- ROC curve visualization (saved to models/reports/roc_curve.png)
- Structured evaluation metrics (saved to models/reports/test_metrics.json)
- Formatted summary report printed to stdout
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless/CLI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from ml.config import (
    CONFUSION_MATRIX_PATH,
    EVALUATION_REPORT_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    REPORTS_DIR,
    ROC_CURVE_PATH,
    TEST_DATA_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path | str = MODEL_PATH) -> Pipeline:
    """Load the trained model pipeline artifact.

    Args:
        model_path: Path to model.joblib file. Defaults to MODEL_PATH.

    Returns:
        Loaded scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If model artifact does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at: {path}. "
            "Please run 'python -m ml.train' first."
        )
    model = joblib.load(path)
    logger.info("Loaded model artifact from: %s", path)
    return model


def load_test_dataset(test_path: Path | str = TEST_DATA_PATH) -> pd.DataFrame:
    """Load the untouched held-out test dataset.

    Args:
        test_path: Path to test.csv file. Defaults to TEST_DATA_PATH.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If test dataset does not exist.
        ValueError: If required columns ('text', 'label') are missing.
    """
    path = Path(test_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Test dataset not found at: {path}. "
            "Please ensure data has been preprocessed and split."
        )
    df = pd.read_csv(path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Test dataset at {path} missing required columns: {missing}")
    logger.info("Loaded untouched test dataset: %d samples from %s", len(df), path)
    return df


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """Compute comprehensive classification performance metrics.

    Args:
        y_true: Ground truth binary labels (0 = genuine, 1 = deceptive).
        y_pred: Predicted binary labels.
        y_score: Continuous prediction scores / probabilities for class 1.

    Returns:
        Dictionary of computed performance metrics.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    counts = np.bincount(y_true, minlength=2)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # ROC-AUC requires at least two classes present in y_true
    roc_auc: Optional[float] = None
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception as e:
            logger.warning("Could not compute ROC-AUC score: %s", e)

    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "per_class": {
            "genuine_0": {
                "precision": round(float(precisions[0]), 4),
                "recall": round(float(recalls[0]), 4),
                "f1": round(float(f1s[0]), 4),
                "support": int(counts[0]),
            },
            "deceptive_1": {
                "precision": round(float(precisions[1]), 4),
                "recall": round(float(recalls[1]), 4),
                "f1": round(float(f1s[1]), 4),
                "support": int(counts[1]),
            },
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path | str = CONFUSION_MATRIX_PATH,
) -> Path:
    """Plot and save confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix array.
        save_path: Path where the PNG image will be saved.

    Returns:
        Path to saved PNG file.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["Genuine (0)", "Deceptive (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    # Annotate counts and percentages
    total = np.sum(cm)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / total * 100.0) if total > 0 else 0.0
            color = "white" if count > thresh else "black"
            ax.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_title("ReviewLens Confusion Matrix (Held-Out Test Set)", fontsize=12, pad=12)
    fig.tight_layout()

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix plot to: %s", path)
    return path


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    roc_auc: Optional[float],
    save_path: Path | str = ROC_CURVE_PATH,
) -> Optional[Path]:
    """Plot and save ROC curve.

    Args:
        y_true: Ground truth binary labels.
        y_score: Continuous prediction scores / probabilities.
        roc_auc: Computed ROC-AUC score.
        save_path: Path where the PNG image will be saved.

    Returns:
        Path to saved PNG file, or None if curve could not be plotted.
    """
    if len(np.unique(y_true)) < 2:
        logger.warning("Single-class ground truth; skipping ROC curve generation.")
        return None

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_label = f"ROC Curve (AUC = {roc_auc:.4f})" if roc_auc is not None else "ROC Curve"

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=auc_label)
    ax.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, labelpad=8)
    ax.set_title("ReviewLens ROC Curve (Held-Out Test Set)", fontsize=12, pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve plot to: %s", path)
    return path


def print_summary_table(
    metrics: dict[str, Any],
    sample_count: int,
    model_name: str = "Trained Pipeline",
) -> None:
    """Print a clean, professional summary table of test set evaluation results.

    Args:
        metrics: Dictionary of computed performance metrics.
        sample_count: Number of test samples evaluated.
        model_name: Descriptive name or identifier of evaluated model.
    """
    c0 = metrics["per_class"]["genuine_0"]
    c1 = metrics["per_class"]["deceptive_1"]
    cm = metrics["confusion_matrix"]
    roc_auc_str = (
        f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A"
    )

    print("\n" + "=" * 70)
    print("                 REVIEWLENS HELD-OUT TEST EVALUATION")
    print("=" * 70)
    print(f"Model: {model_name} (v{MODEL_VERSION})")
    print(f"Untouched Test Samples: {sample_count} (Genuine: {c0['support']}, Deceptive: {c1['support']})")
    print("-" * 70)
    print(f"{'Class / Metric':<24} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support'}")
    print("-" * 70)
    print(f"{'Genuine (0)':<24} | {c0['precision']:<10.4f} | {c0['recall']:<10.4f} | {c0['f1']:<10.4f} | {c0['support']}")
    print(f"{'Deceptive (1)':<24} | {c1['precision']:<10.4f} | {c1['recall']:<10.4f} | {c1['f1']:<10.4f} | {c1['support']}")
    print("-" * 70)
    print(f"{'Macro Average':<24} | {'-':<10} | {'-':<10} | {metrics['macro_f1']:<10.4f} | {sample_count}")
    print(f"{'Overall Accuracy':<24} | {metrics['accuracy']:<10.4f}")
    print(f"{'ROC-AUC Score':<24} | {roc_auc_str}")
    print("=" * 70)

    print("\nConfusion Matrix Breakdown:")
    print(f"  True Negatives  (Genuine correctly classified):   {cm['tn']}")
    print(f"  False Positives (Genuine misclassified as fake):  {cm['fp']}")
    print(f"  False Negatives (Deceptive missed as genuine):    {cm['fn']}")
    print(f"  True Positives  (Deceptive correctly detected):   {cm['tp']}")
    print("=" * 70 + "\n")


def evaluate(
    model_path: Path | str = MODEL_PATH,
    test_path: Path | str = TEST_DATA_PATH,
    reports_dir: Path | str = REPORTS_DIR,
    confusion_matrix_path: Path | str = CONFUSION_MATRIX_PATH,
    roc_curve_path: Path | str = ROC_CURVE_PATH,
    report_json_path: Path | str = EVALUATION_REPORT_PATH,
) -> dict[str, Any]:
    """Execute end-to-end evaluation on the untouched test set once per run.

    Args:
        model_path: Path to model.joblib.
        test_path: Path to test.csv.
        reports_dir: Output directory for evaluation reports and plots.
        confusion_matrix_path: Target path for confusion_matrix.png.
        roc_curve_path: Target path for roc_curve.png.
        report_json_path: Target path for test_metrics.json.

    Returns:
        Dictionary containing all evaluation metrics and generated report paths.
    """
    logger.info("Starting ReviewLens test evaluation...")

    # Ensure output directory exists
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load model and test dataset
    model = load_model(model_path)
    test_df = load_test_dataset(test_path)

    X_test = test_df["text"]
    y_test = np.array(test_df["label"].values, dtype=int)

    # 2. Evaluate model on untouched test set exactly once per run
    logger.info("Predicting on %d test samples...", len(test_df))
    y_pred = model.predict(X_test)

    # Extract continuous scores (probabilities or decision function) for ROC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    # 3. Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_score)

    # 4. Generate and save visualizations
    cm_array = np.array(metrics["confusion_matrix"]["matrix"])
    cm_plot_path = plot_confusion_matrix(cm_array, save_path=confusion_matrix_path)
    roc_plot_path = plot_roc_curve(y_test, y_score, metrics["roc_auc"], save_path=roc_curve_path)

    # 5. Save structured evaluation report JSON
    report_data = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "test_sample_count": len(test_df),
        "metrics": metrics,
        "artifacts": {
            "confusion_matrix_png": str(cm_plot_path),
            "roc_curve_png": str(roc_plot_path) if roc_plot_path else None,
        },
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Saved evaluation report JSON to: %s", report_json_path)

    # 6. Print clean summary table
    clf_step = (
        model.named_steps["classifier"].__class__.__name__
        if hasattr(model, "named_steps") and "classifier" in model.named_steps
        else "Classifier"
    )
    print_summary_table(metrics, len(test_df), model_name=clf_step)

    logger.info("Evaluation finished successfully.")
    return report_data


def main() -> None:
    """CLI entrypoint for ml/evaluate.py."""
    parser = argparse.ArgumentParser(
        description="Evaluate the saved ReviewLens model on the held-out test set."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to trained model.joblib (default: models/model.joblib)",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=TEST_DATA_PATH,
        help="Path to test CSV file (default: data/processed/test.csv)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory to save evaluation reports and plots (default: models/reports/)",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        test_path=args.test_path,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
