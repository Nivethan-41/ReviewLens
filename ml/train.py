"""Model training and calibration pipeline for ReviewLens fake review detection.

Loads processed train and validation datasets, constructs full scikit-learn
pipelines combining feature extraction and candidate classifiers, evaluates
them using 5-fold stratified cross-validation on train and held-out validation macro-F1,
performs hyperparameter tuning on the top two candidate models, selects the best model,
runs an ablation study, and saves the trained pipeline and training metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.config import (
    FEATURES_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    RANDOM_SEED,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from ml.features import build_feature_pipeline, save_feature_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Tie-breaking priority order: lower rank = simpler, more explainable model
SIMPLICITY_ORDER: dict[str, int] = {
    "LogisticRegression": 1,
    "LinearSVC": 2,
    "HistGradientBoosting": 3,
    "RandomForest": 4,
}

# Hyperparameter search distributions for top models
HYPERPARAMETER_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "LogisticRegression": {
        "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
    },
    "LinearSVC": {
        "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    },
    "RandomForest": {
        "classifier__n_estimators": [50, 100, 150, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    },
    "HistGradientBoosting": {
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__max_iter": [50, 100, 150],
    },
}


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load and validate dataset from a CSV file.

    Args:
        path: Path to dataset CSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If dataset path does not exist.
        ValueError: If 'text' or 'label' columns are missing.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)
    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset at {file_path} missing required columns: {missing}"
        )
    return df


def build_pipeline_for_classifier(
    classifier: Any,
    include_word: bool = True,
    include_char: bool = True,
    include_stylometric: bool = True,
) -> Pipeline:
    """Create a full scikit-learn Pipeline with feature extraction and a classifier.

    Args:
        classifier: Estimator instance implementing fit and predict.
        include_word: Whether to include word TF-IDF features. Defaults to True.
        include_char: Whether to include character TF-IDF features. Defaults to True.
        include_stylometric: Whether to include stylometric features. Defaults to True.

    Returns:
        Full scikit-learn Pipeline.
    """
    features = build_feature_pipeline(
        include_word=include_word,
        include_char=include_char,
        include_stylometric=include_stylometric,
    )
    return Pipeline([
        ("features", features),
        ("classifier", classifier),
    ])


def check_feature_density(X_train: pd.Series | list[str]) -> tuple[bool, float]:
    """Inspect the feature matrix density on the training data.

    Args:
        X_train: Review texts from the training dataset.

    Returns:
        Tuple of (is_dense_enough, density_ratio).
    """
    sample_feat_pipe = build_feature_pipeline()
    transformed = sample_feat_pipe.fit_transform(X_train)
    if sparse.issparse(transformed):
        total_elements = transformed.shape[0] * transformed.shape[1]
        density = (
            float(transformed.nnz) / float(total_elements)
            if total_elements > 0
            else 0.0
        )
        # Sparse matrices in text classification typically have density < 0.1
        # HistGradientBoosting requires a dense ndarray
        is_dense = False
    else:
        is_dense = True
        density = 1.0

    return is_dense, density


def instantiate_candidate_classifiers(
    random_seed: int = RANDOM_SEED,
    feature_is_dense: bool = False,
    density: float = 0.0,
) -> dict[str, Any]:
    """Instantiate candidate classifiers according to specification.

    Args:
        random_seed: Random state seed.
        feature_is_dense: Whether the feature representation is dense.
        density: Density ratio of feature matrix.

    Returns:
        Dictionary mapping model names to classifier instances.
    """
    classifiers: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            random_state=random_seed,
        ),
        "LinearSVC": LinearSVC(
            class_weight="balanced",
            random_state=random_seed,
            dual="auto",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }

    if feature_is_dense:
        classifiers["HistGradientBoosting"] = HistGradientBoostingClassifier(
            random_state=random_seed,
        )
    else:
        logger.info(
            "Skipping HistGradientBoostingClassifier: feature matrix is sparse "
            "(density=%.4f, dense matrix required by HistGradientBoosting).",
            density,
        )

    return classifiers


def evaluate_candidate_models(
    classifiers: dict[str, Any],
    X_train: pd.Series | list[str],
    y_train: pd.Series | np.ndarray,
    X_val: pd.Series | list[str],
    y_val: pd.Series | np.ndarray,
    random_seed: int = RANDOM_SEED,
    cv_splits: int = 5,
) -> dict[str, dict[str, Any]]:
    """Train and evaluate candidate models using 5-fold CV on train and validation set.

    Args:
        classifiers: Mapping of model name to unfitted classifier.
        X_train: Training text features.
        y_train: Training labels.
        X_val: Validation text features.
        y_val: Validation labels.
        random_seed: Random seed.
        cv_splits: Number of stratified cross-validation folds.

    Returns:
        Dictionary mapping model names to evaluation results.
    """
    # Safeguard cv_splits against very small datasets
    min_class_count = min(np.bincount(np.array(y_train, dtype=int)))
    effective_cv_splits = max(2, min(cv_splits, min_class_count))

    cv = StratifiedKFold(
        n_splits=effective_cv_splits,
        shuffle=True,
        random_state=random_seed,
    )

    results: dict[str, dict[str, Any]] = {}

    for name, clf in classifiers.items():
        logger.info("Evaluating candidate model: %s...", name)
        pipeline = build_pipeline_for_classifier(clf)

        # 5-fold Stratified CV on training set
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Train on full train set and evaluate on val set
        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)

        val_macro_f1 = float(f1_score(y_val, y_val_pred, average="macro"))
        val_accuracy = float(accuracy_score(y_val, y_val_pred))

        logger.info(
            "[%s] CV Macro-F1: %.4f (+/- %.4f) | Val Macro-F1: %.4f | Val Acc: %.4f",
            name,
            cv_mean,
            cv_std,
            val_macro_f1,
            val_accuracy,
        )

        results[name] = {
            "pipeline": pipeline,
            "cv_macro_f1_mean": round(cv_mean, 4),
            "cv_macro_f1_std": round(cv_std, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_accuracy": round(val_accuracy, 4),
            "tuned": False,
            "best_params": None,
        }

    return results


def tune_top_models(
    results: dict[str, dict[str, Any]],
    X_train: pd.Series | list[str],
    y_train: pd.Series | np.ndarray,
    X_val: pd.Series | list[str],
    y_val: pd.Series | np.ndarray,
    n_iter: int = 5,
    random_seed: int = RANDOM_SEED,
    cv_splits: int = 5,
) -> dict[str, dict[str, Any]]:
    """Perform time-boxed hyperparameter search for the top two candidate models.

    Args:
        results: Candidate models dictionary from evaluate_candidate_models.
        X_train: Training texts.
        y_train: Training labels.
        X_val: Validation texts.
        y_val: Validation labels.
        n_iter: Maximum iterations for RandomizedSearchCV.
        random_seed: Random seed.
        cv_splits: Number of stratified folds.

    Returns:
        Updated results dictionary including tuned models and parameters.
    """
    # Sort models by val macro-F1 (descending), breaking ties by CV macro-F1
    ranked = sorted(
        results.keys(),
        key=lambda m: (
            results[m]["val_macro_f1"],
            results[m]["cv_macro_f1_mean"],
        ),
        reverse=True,
    )
    top_two = ranked[:2]
    logger.info("Top two candidate models selected for hyperparameter tuning: %s", top_two)

    min_class_count = min(np.bincount(np.array(y_train, dtype=int)))
    effective_cv_splits = max(2, min(cv_splits, min_class_count))
    cv = StratifiedKFold(
        n_splits=effective_cv_splits,
        shuffle=True,
        random_state=random_seed,
    )

    for name in top_two:
        param_dist = HYPERPARAMETER_DISTRIBUTIONS.get(name)
        if not param_dist:
            continue

        pipeline = results[name]["pipeline"]
        total_combos = len(list(ParameterGrid(param_dist)))
        effective_n_iter = max(1, min(n_iter, total_combos))

        logger.info(
            "Tuning %s with RandomizedSearchCV (n_iter=%d of %d combos)...",
            name,
            effective_n_iter,
            total_combos,
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=effective_n_iter,
            cv=cv,
            scoring="f1_macro",
            random_state=random_seed,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        tuned_pipeline = search.best_estimator_
        y_val_tuned = tuned_pipeline.predict(X_val)
        val_macro_f1_tuned = float(f1_score(y_val, y_val_tuned, average="macro"))
        val_acc_tuned = float(accuracy_score(y_val, y_val_tuned))

        logger.info(
            "[%s Tuned] Best CV Score: %.4f | Val Macro-F1: %.4f (Untuned: %.4f) | Best Params: %s",
            name,
            search.best_score_,
            val_macro_f1_tuned,
            results[name]["val_macro_f1"],
            search.best_params_,
        )

        # Update if tuned model is equal or better on validation
        if val_macro_f1_tuned >= results[name]["val_macro_f1"]:
            results[name]["pipeline"] = tuned_pipeline
            results[name]["cv_macro_f1_mean"] = round(float(search.best_score_), 4)
            results[name]["val_macro_f1"] = round(val_macro_f1_tuned, 4)
            results[name]["val_accuracy"] = round(val_acc_tuned, 4)
            results[name]["tuned"] = True
            results[name]["best_params"] = search.best_params_

    return results


def select_best_model(
    results: dict[str, dict[str, Any]],
) -> tuple[str, Pipeline]:
    """Select the best model based on validation macro-F1 with explainability tie-breaking.

    Ties go to the simpler, more explainable model according to SIMPLICITY_ORDER.

    Args:
        results: Dictionary of model evaluation results.

    Returns:
        Tuple of (chosen_model_name, chosen_pipeline).
    """
    def ranking_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, int]:
        name, res = item
        # Negate metrics for descending order; positive integer for simplicity order
        return (
            -round(res["val_macro_f1"], 4),
            -round(res["cv_macro_f1_mean"], 4),
            SIMPLICITY_ORDER.get(name, 99),
        )

    best_item = min(results.items(), key=ranking_key)
    chosen_name = best_item[0]
    chosen_pipeline = best_item[1]["pipeline"]

    logger.info(
        "Chosen best model: '%s' (Val Macro-F1: %.4f, Simplicity Rank: %d)",
        chosen_name,
        best_item[1]["val_macro_f1"],
        SIMPLICITY_ORDER.get(chosen_name, 99),
    )
    return chosen_name, chosen_pipeline


def run_feature_ablation_study(
    chosen_classifier: Any,
    X_train: pd.Series | list[str],
    y_train: pd.Series | np.ndarray,
    X_val: pd.Series | list[str],
    y_val: pd.Series | np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Run feature ablation study measuring validation macro-F1 across feature subsets.

    Ablation setups evaluated:
    (a) Word TF-IDF only
    (b) Word TF-IDF + Char TF-IDF
    (c) Word TF-IDF + Char TF-IDF + Stylometric features (Full)

    Args:
        chosen_classifier: Classifier instance to use for ablation.
        X_train: Training texts.
        y_train: Training labels.
        X_val: Validation texts.
        y_val: Validation labels.

    Returns:
        Dictionary of ablation results.
    """
    logger.info("Running feature ablation study...")

    ablation_configs = [
        (
            "word_tfidf_only",
            {"include_word": True, "include_char": False, "include_stylometric": False},
            ["word_tfidf"],
        ),
        (
            "word_plus_char_tfidf",
            {"include_word": True, "include_char": True, "include_stylometric": False},
            ["word_tfidf", "char_tfidf"],
        ),
        (
            "full_features",
            {"include_word": True, "include_char": True, "include_stylometric": True},
            ["word_tfidf", "char_tfidf", "stylometric"],
        ),
    ]

    ablation_results: dict[str, dict[str, Any]] = {}

    for name, flags, feature_list in ablation_configs:
        # Clone classifier by creating new instance with same parameters
        clf_class = chosen_classifier.__class__
        clf_instance = clf_class(**chosen_classifier.get_params())

        pipe = build_pipeline_for_classifier(clf_instance, **flags)
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_val)
        val_macro_f1 = float(f1_score(y_val, preds, average="macro"))
        val_acc = float(accuracy_score(y_val, preds))

        logger.info(
            "Ablation [%s]: Val Macro-F1 = %.4f | Val Accuracy = %.4f",
            name,
            val_macro_f1,
            val_acc,
        )

        ablation_results[name] = {
            "features": feature_list,
            "val_macro_f1": round(val_macro_f1, 4),
            "val_accuracy": round(val_acc, 4),
        }

    return ablation_results


def save_model(pipeline: Pipeline, path: Path | str = MODEL_PATH) -> Path:
    """Save trained pipeline artifact to disk using joblib.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        path: Target file path. Defaults to MODEL_PATH.

    Returns:
        Path to saved artifact.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info("Saved model artifact to: %s", save_path)
    return save_path


def save_metrics(metrics: dict[str, Any], path: Path | str = METRICS_PATH) -> Path:
    """Save metrics JSON artifact to disk.

    Args:
        metrics: Metrics dictionary.
        path: Target file path. Defaults to METRICS_PATH.

    Returns:
        Path to saved JSON file.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics artifact to: %s", save_path)
    return save_path


def print_comparison_table(
    results: dict[str, dict[str, Any]],
    chosen_model: str,
    ablation: dict[str, dict[str, Any]],
) -> None:
    """Print readable summary tables for model comparison and ablation results.

    Args:
        results: Dictionary of candidate model results.
        chosen_model: Name of chosen best model.
        ablation: Dictionary of ablation study results.
    """
    print("\n" + "=" * 78)
    print("                      REVIEWLENS MODEL COMPARISON")
    print("=" * 78)
    header = f"{'Model':<24} | {'CV Macro-F1 (std)':<20} | {'Val Macro-F1':<14} | {'Val Acc':<8} | {'Best'}"
    print(header)
    print("-" * 78)

    for name, res in results.items():
        is_best = "*" if name == chosen_model else ""
        cv_str = f"{res['cv_macro_f1_mean']:.4f} (+/- {res['cv_macro_f1_std']:.4f})"
        val_f1_str = f"{res['val_macro_f1']:.4f}"
        val_acc_str = f"{res['val_accuracy']:.4f}"
        print(f"{name:<24} | {cv_str:<20} | {val_f1_str:<14} | {val_acc_str:<8} | {is_best}")

    print("=" * 78)
    print(f"* Chosen model: {chosen_model} (ties go to simpler, explainable model)\n")

    print("-" * 78)
    print("                      FEATURE ABLATION STUDY")
    print("-" * 78)
    ablation_header = f"{'Ablation Setup':<28} | {'Features Included':<30} | {'Val Macro-F1'}"
    print(ablation_header)
    print("-" * 78)

    for setup_name, a_res in ablation.items():
        features_str = ", ".join(a_res["features"])
        print(f"{setup_name:<28} | {features_str:<30} | {a_res['val_macro_f1']:.4f}")
    print("=" * 78 + "\n")


def train(
    train_path: Path | str = TRAIN_DATA_PATH,
    val_path: Path | str = VAL_DATA_PATH,
    test_path: Path | str = TEST_DATA_PATH,
    model_save_path: Path | str = MODEL_PATH,
    metrics_save_path: Path | str = METRICS_PATH,
    n_iter: int = 5,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Execute end-to-end training, hyperparameter tuning, model selection, and artifact saving.

    Args:
        train_path: Path to train.csv.
        val_path: Path to val.csv.
        test_path: Path to test.csv (for row count record).
        model_save_path: Destination for model.joblib.
        metrics_save_path: Destination for metrics.json.
        n_iter: Number of iterations for hyperparameter search.
        random_seed: Random seed.

    Returns:
        Complete metrics dictionary saved to disk.
    """
    logger.info("Starting ReviewLens model training pipeline...")

    # 1. Load datasets
    train_df = load_dataset(train_path)
    val_df = load_dataset(val_path)

    test_rows: Optional[int] = None
    if Path(test_path).exists():
        try:
            test_df = load_dataset(test_path)
            test_rows = len(test_df)
        except Exception:
            test_rows = None

    dataset_rows = {
        "train": len(train_df),
        "val": len(val_df),
        "test": test_rows,
    }
    logger.info(
        "Dataset row counts - Train: %d, Val: %d, Test: %s",
        dataset_rows["train"],
        dataset_rows["val"],
        dataset_rows["test"],
    )

    X_train, y_train = train_df["text"], train_df["label"].values
    X_val, y_val = val_df["text"], val_df["label"].values

    # 2. Check feature density to determine if HistGradientBoosting is supported
    is_dense, density = check_feature_density(X_train)

    # 3. Instantiate classifiers
    classifiers = instantiate_candidate_classifiers(
        random_seed=random_seed,
        feature_is_dense=is_dense,
        density=density,
    )

    # 4. 5-Fold Stratified CV on Train + Val evaluation
    results = evaluate_candidate_models(
        classifiers=classifiers,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_seed=random_seed,
        cv_splits=5,
    )

    # 5. Hyperparameter search on top two models
    results = tune_top_models(
        results=results,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_iter=n_iter,
        random_seed=random_seed,
        cv_splits=5,
    )

    # 6. Select best model by validation macro-F1 (ties broken by simplicity)
    chosen_name, chosen_pipeline = select_best_model(results)

    # 7. Ablation study on feature branches using the chosen classifier type
    chosen_clf_estimator = chosen_pipeline.named_steps["classifier"]
    ablation_results = run_feature_ablation_study(
        chosen_classifier=chosen_clf_estimator,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    # 8. Save model pipeline and features pipeline
    save_model(chosen_pipeline, model_save_path)
    if "features" in chosen_pipeline.named_steps:
        save_feature_pipeline(
            chosen_pipeline.named_steps["features"],
            FEATURES_PATH,
        )

    # 9. Format and save metrics.json
    model_comparisons_serializable: dict[str, Any] = {}
    for name, r in results.items():
        model_comparisons_serializable[name] = {
            "cv_macro_f1_mean": r["cv_macro_f1_mean"],
            "cv_macro_f1_std": r["cv_macro_f1_std"],
            "val_macro_f1": r["val_macro_f1"],
            "val_accuracy": r["val_accuracy"],
            "tuned": r["tuned"],
            "best_params": (
                {k: str(v) for k, v in r["best_params"].items()}
                if r["best_params"]
                else None
            ),
            "is_best": (name == chosen_name),
        }

    metrics_data: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "chosen_model": chosen_name,
        "chosen_val_macro_f1": results[chosen_name]["val_macro_f1"],
        "dataset_rows": dataset_rows,
        "model_comparisons": model_comparisons_serializable,
        "ablation": ablation_results,
    }

    save_metrics(metrics_data, metrics_save_path)

    # 10. Print formatted tables
    print_comparison_table(results, chosen_name, ablation_results)

    logger.info("Training pipeline completed successfully.")
    return metrics_data


def main() -> None:
    """CLI entrypoint for ml/train.py."""
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and tune ReviewLens fake review detection models."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=TRAIN_DATA_PATH,
        help="Path to training CSV file (default: data/processed/train.csv)",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=VAL_DATA_PATH,
        help="Path to validation CSV file (default: data/processed/val.csv)",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=TEST_DATA_PATH,
        help="Path to test CSV file (default: data/processed/test.csv)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=5,
        help="Number of iterations for hyperparameter search (default: 5)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})",
    )
    args = parser.parse_args()

    train(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        n_iter=args.n_iter,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
