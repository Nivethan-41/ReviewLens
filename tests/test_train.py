"""Unit and integration tests for ReviewLens model training pipeline (ml/train.py)."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml.config import MODEL_VERSION, RANDOM_SEED
from ml.train import (
    SIMPLICITY_ORDER,
    check_feature_density,
    instantiate_candidate_classifiers,
    load_dataset,
    run_feature_ablation_study,
    select_best_model,
    train,
)


@pytest.fixture
def synthetic_40_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Generate a 40-row synthetic dataset with train, val, and test splits."""
    genuine_texts = [
        f"I bought this product {i} weeks ago. It functions as expected with good build quality."
        for i in range(20)
    ]
    deceptive_texts = [
        f"AMAZING UNBELIEVABLE BEST EVER INCREDIBLE 100% PERFECT MUST BUY NOW WOW {i}!!!"
        for i in range(20)
    ]

    df = pd.DataFrame({
        "text": genuine_texts + deceptive_texts,
        "label": [0] * 20 + [1] * 20,
        "rating": [4] * 20 + [5] * 20,
    })

    # Shuffle with fixed seed
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # 40 rows: 28 train, 6 val, 6 test
    train_df = df.iloc[:28].copy()
    val_df = df.iloc[28:34].copy()
    test_df = df.iloc[34:].copy()

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def test_train_end_to_end_synthetic_40_rows(
    synthetic_40_dataset: tuple[Path, Path, Path], tmp_path: Path
):
    """Fast integration test: train on 40-row synthetic dataset and verify artifacts and keys."""
    train_csv, val_csv, test_csv = synthetic_40_dataset
    model_out = tmp_path / "model.joblib"
    metrics_out = tmp_path / "metrics.json"

    metrics = train(
        train_path=train_csv,
        val_path=val_csv,
        test_path=test_csv,
        model_save_path=model_out,
        metrics_save_path=metrics_out,
        n_iter=3,
        random_seed=RANDOM_SEED,
    )

    # Verify artifacts exist
    assert model_out.exists(), "model.joblib was not created"
    assert metrics_out.exists(), "metrics.json was not created"

    # Verify model artifact is loadable and functional
    pipeline = joblib.load(model_out)
    assert isinstance(pipeline, Pipeline)
    assert "features" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    preds = pipeline.predict(["This is an ordinary review text with normal words."])
    assert len(preds) == 1
    assert preds[0] in {0, 1}

    # Verify metrics.json has expected keys
    with open(metrics_out, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)

    expected_top_keys = {
        "model_version",
        "training_date",
        "chosen_model",
        "chosen_val_macro_f1",
        "dataset_rows",
        "model_comparisons",
        "ablation",
    }
    assert expected_top_keys.issubset(saved_metrics.keys())
    assert saved_metrics["model_version"] == MODEL_VERSION
    assert saved_metrics["dataset_rows"]["train"] == 28
    assert saved_metrics["dataset_rows"]["val"] == 6
    assert saved_metrics["dataset_rows"]["test"] == 6

    # Verify model comparisons structure
    comparisons = saved_metrics["model_comparisons"]
    assert "LogisticRegression" in comparisons
    assert "LinearSVC" in comparisons
    assert "RandomForest" in comparisons

    for model_name, info in comparisons.items():
        assert "cv_macro_f1_mean" in info
        assert "cv_macro_f1_std" in info
        assert "val_macro_f1" in info
        assert "val_accuracy" in info
        assert "is_best" in info
        assert 0.0 <= info["val_macro_f1"] <= 1.0

    # Verify ablation keys
    ablation = saved_metrics["ablation"]
    assert "word_tfidf_only" in ablation
    assert "word_plus_char_tfidf" in ablation
    assert "full_features" in ablation
    assert ablation["word_tfidf_only"]["features"] == ["word_tfidf"]
    assert ablation["word_plus_char_tfidf"]["features"] == ["word_tfidf", "char_tfidf"]
    assert ablation["full_features"]["features"] == [
        "word_tfidf",
        "char_tfidf",
        "stylometric",
    ]


def test_select_best_model_tie_breaking():
    """Verify that ties go to the simpler, more explainable model."""
    results = {
        "RandomForest": {
            "pipeline": Pipeline([("step", None)]),
            "val_macro_f1": 0.95,
            "cv_macro_f1_mean": 0.92,
        },
        "LogisticRegression": {
            "pipeline": Pipeline([("step", None)]),
            "val_macro_f1": 0.95,
            "cv_macro_f1_mean": 0.92,
        },
        "LinearSVC": {
            "pipeline": Pipeline([("step", None)]),
            "val_macro_f1": 0.95,
            "cv_macro_f1_mean": 0.92,
        },
    }

    chosen_name, _ = select_best_model(results)
    assert chosen_name == "LogisticRegression", "Tie-breaking should favor LogisticRegression"


def test_skip_hist_gradient_boosting_on_sparse():
    """Verify HistGradientBoosting is skipped when features are sparse."""
    classifiers = instantiate_candidate_classifiers(
        random_seed=RANDOM_SEED,
        feature_is_dense=False,
        density=0.04,
    )
    assert "HistGradientBoosting" not in classifiers
    assert "LogisticRegression" in classifiers
    assert "LinearSVC" in classifiers
    assert "RandomForest" in classifiers


def test_load_dataset_missing_columns(tmp_path: Path):
    """Verify load_dataset raises ValueError when required columns are missing."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"wrong_col": [1, 2]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(bad_csv)
