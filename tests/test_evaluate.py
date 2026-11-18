"""Unit and integration tests for ReviewLens model evaluation module (ml/evaluate.py)."""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.evaluate import (
    compute_metrics,
    evaluate,
    load_model,
    load_test_dataset,
    plot_confusion_matrix,
    plot_roc_curve,
)
from ml.features import build_feature_pipeline


@pytest.fixture
def trained_pipeline_and_test_data(tmp_path: Path) -> tuple[Path, Path]:
    """Train a simple pipeline and save it alongside a synthetic test set."""
    texts = [
        "This is an honest review of a real product that works reasonably well.",
        "Good product overall, fast shipping, would consider buying again.",
        "AMAZING BEST EVER INCREDIBLE 100% PERFECT MUST BUY NOW WOW!",
        "UNBELIEVABLE MIRACLE PRODUCT BEST EXPERIENCE EVER BUY IMMEDIATELY!",
        "Okay product, meets expectations for the low price.",
        "SUPERLATIVE EXTRAVAGANZA BEST OF THE BEST TOP RATED SUPER!",
    ]
    labels = [0, 0, 1, 1, 0, 1]

    pipe = Pipeline([
        ("features", build_feature_pipeline()),
        ("classifier", LogisticRegression(random_state=42, solver="liblinear")),
    ])
    pipe.fit(texts, labels)

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipe, model_path)

    test_df = pd.DataFrame({
        "text": [
            "Normal review describing standard usage and decent results.",
            "AMAZING GREATEST ITEM EVER IN THE WORLD BUY NOW!",
        ],
        "label": [0, 1],
    })
    test_path = tmp_path / "test.csv"
    test_df.to_csv(test_path, index=False)

    return model_path, test_path


def test_compute_metrics_accuracy_and_f1():
    """Verify metric computation on deterministic binary predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])  # 1 FP, 0 FN -> accuracy = 3/4 = 0.75
    y_score = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_score)

    assert metrics["accuracy"] == 0.75
    assert "macro_f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["confusion_matrix"]["tn"] == 1
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 0
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["per_class"]["genuine_0"]["support"] == 2
    assert metrics["per_class"]["deceptive_1"]["support"] == 2


def test_evaluate_end_to_end_artifacts(
    trained_pipeline_and_test_data: tuple[Path, Path], tmp_path: Path
):
    """Verify evaluate() runs once per run, outputs plots and reports JSON."""
    model_path, test_path = trained_pipeline_and_test_data
    reports_dir = tmp_path / "reports"
    cm_path = reports_dir / "confusion_matrix.png"
    roc_path = reports_dir / "roc_curve.png"
    json_path = reports_dir / "test_metrics.json"

    report = evaluate(
        model_path=model_path,
        test_path=test_path,
        reports_dir=reports_dir,
        confusion_matrix_path=cm_path,
        roc_curve_path=roc_path,
        report_json_path=json_path,
    )

    # Verify return dictionary structure
    assert "metrics" in report
    assert "artifacts" in report
    assert report["test_sample_count"] == 2

    # Verify artifacts exist on disk
    assert cm_path.exists(), "confusion_matrix.png was not generated"
    assert roc_path.exists(), "roc_curve.png was not generated"
    assert json_path.exists(), "test_metrics.json was not generated"

    # Verify JSON content
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["metrics"]["accuracy"] >= 0.0
    assert "per_class" in loaded["metrics"]


def test_plot_helpers_create_valid_files(tmp_path: Path):
    """Verify plot_confusion_matrix and plot_roc_curve create valid image files."""
    cm = np.array([[10, 2], [1, 15]])
    cm_target = tmp_path / "cm.png"
    saved_cm = plot_confusion_matrix(cm, save_path=cm_target)
    assert saved_cm.exists()
    assert saved_cm.stat().st_size > 0

    roc_target = tmp_path / "roc.png"
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.8, 0.9])
    saved_roc = plot_roc_curve(y_true, y_score, roc_auc=0.95, save_path=roc_target)
    assert saved_roc is not None
    assert saved_roc.exists()
    assert saved_roc.stat().st_size > 0
