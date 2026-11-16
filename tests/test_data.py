"""Tests for ReviewLens data ingestion, cleaning, and splitting module (ml/data.py)."""

import sys
import subprocess
from pathlib import Path
import pandas as pd
import pytest

from ml.config import RANDOM_SEED, TEST_SIZE, VAL_SIZE
from ml.data import COLUMN_MAPPING, clean_dataset, load_raw, split_dataset


def test_load_raw_label_mapping(tmp_path: Path):
    """Verify that load_raw correctly maps diverse label formats to 0 and 1."""
    # Test with OR / CG and custom column names
    data = {
        "text_": [
            "This is a genuine product that works very well indeed.",
            "I am writing a fake review generated to fool algorithms.",
            "Another real review for a great hotel stay experience.",
            "Another deceptive computer generated review that is clearly artificial.",
        ],
        "label": ["OR", "CG", "truthful", "deceptive"],
        "rating": [5, 1, 4, 5],
        "category": ["electronics", "electronics", "hotels", "hotels"],
    }
    raw_csv = tmp_path / "raw_test.csv"
    pd.DataFrame(data).to_csv(raw_csv, index=False)

    df_loaded = load_raw(raw_csv)

    assert list(df_loaded.columns) == ["text", "label", "rating", "category"]
    assert df_loaded["label"].tolist() == [0, 1, 0, 1]
    assert df_loaded["label"].dtype.kind in ("i", "u")


def test_load_raw_custom_mapping(tmp_path: Path):
    """Verify that custom column mapping can be supplied to load_raw."""
    data = {
        "my_review": ["A wonderful dining experience with great food and drinks."],
        "my_target": ["real"],
        "stars": [5],
    }
    raw_csv = tmp_path / "custom.csv"
    pd.DataFrame(data).to_csv(raw_csv, index=False)

    custom_mapping = {
        "text_col": "my_review",
        "label_col": "my_target",
        "rating_col": "stars",
        "category_col": "cat",
        "label_map": {"real": 0, "fake": 1},
    }

    df_loaded = load_raw(raw_csv, mapping=custom_mapping)
    assert "text" in df_loaded.columns
    assert "label" in df_loaded.columns
    assert "rating" in df_loaded.columns
    assert df_loaded["label"].iloc[0] == 0


def test_load_raw_missing_file():
    """Verify FileNotFoundError is raised when raw CSV file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_raw("non_existent_file_path_123.csv")


def test_load_raw_missing_columns(tmp_path: Path):
    """Verify KeyError is raised when required columns cannot be found."""
    raw_csv = tmp_path / "bad_cols.csv"
    pd.DataFrame({"unrelated_col": [1, 2, 3]}).to_csv(raw_csv, index=False)

    with pytest.raises(KeyError):
        load_raw(raw_csv)


def test_load_raw_unrecognized_label(tmp_path: Path):
    """Verify ValueError is raised when encountering an unmapped label value."""
    raw_csv = tmp_path / "bad_label.csv"
    pd.DataFrame({"text_": ["Sample review with plenty of words here."], "label": ["unknown_class"]}).to_csv(
        raw_csv, index=False
    )

    with pytest.raises(ValueError, match="Unrecognized label value"):
        load_raw(raw_csv)


def test_clean_dataset_duplicate_removal():
    """Verify that exact duplicate texts are removed, keeping the first occurrence."""
    df = pd.DataFrame(
        {
            "text": [
                "This is the first occurrence of the review text.",
                "This is the first occurrence of the review text.",  # duplicate
                "This is a distinct and unique review text here.",
                "This is the first occurrence of the review text.",  # duplicate
            ],
            "label": [0, 0, 1, 1],
        }
    )

    cleaned_df, report = clean_dataset(df)

    assert len(cleaned_df) == 2
    assert report["rows_before"] == 4
    assert report["rows_after"] == 2
    assert report["duplicates_removed"] == 2
    assert cleaned_df["text"].iloc[0] == "This is the first occurrence of the review text."
    assert cleaned_df["text"].iloc[1] == "This is a distinct and unique review text here."


def test_clean_dataset_short_review_removal():
    """Verify that reviews with fewer than 5 words are dropped."""
    df = pd.DataFrame(
        {
            "text": [
                "Too short",  # 2 words -> drop
                "One two three",  # 3 words -> drop
                "One two three four",  # 4 words -> drop
                "One two three four five",  # 5 words -> keep
                "This review has more than five words in it.",  # 9 words -> keep
            ],
            "label": [0, 1, 0, 1, 0],
        }
    )

    cleaned_df, report = clean_dataset(df)

    assert len(cleaned_df) == 2
    assert report["short_reviews_removed"] == 3
    assert all(len(t.split()) >= 5 for t in cleaned_df["text"])


def test_clean_dataset_null_and_empty():
    """Verify that null, empty, and whitespace-only reviews are removed."""
    df = pd.DataFrame(
        {
            "text": [
                None,
                "",
                "     ",
                "A completely valid review text with enough words to keep.",
            ],
            "label": [0, 1, 0, 1],
        }
    )

    cleaned_df, report = clean_dataset(df)

    assert len(cleaned_df) == 1
    assert report["null_or_empty_removed"] == 3
    assert cleaned_df["text"].iloc[0] == "A completely valid review text with enough words to keep."


def test_clean_dataset_report_class_balance():
    """Verify class balance dictionary accurately reports class counts and proportions."""
    df = pd.DataFrame(
        {
            "text": [
                "Review number one with enough words.",
                "Review number two with enough words.",
                "Review number three with enough words.",
                "Review number four with enough words.",
            ],
            "label": [0, 0, 0, 1],
        }
    )

    cleaned_df, report = clean_dataset(df)

    assert report["class_balance"] == {0: 3, 1: 1}
    assert report["class_balance_pct"] == {0: 0.75, 1: 0.25}


def test_split_dataset_sizes_and_files(tmp_path: Path):
    """Verify that split_dataset creates stratified splits of expected sizes and saves CSVs."""
    # Create synthetic dataset with 100 samples (70 class 0, 30 class 1)
    texts = [f"This is unique synthetic review sentence number {i} for testing." for i in range(100)]
    labels = [0] * 70 + [1] * 30
    df = pd.DataFrame({"text": texts, "label": labels})

    train_df, val_df, test_df = split_dataset(
        df,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        output_dir=tmp_path,
    )

    # Check files exist
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "val.csv").exists()
    assert (tmp_path / "test.csv").exists()

    # Total rows must sum to 100
    assert len(train_df) + len(val_df) + len(test_df) == 100
    assert len(test_df) == 10
    assert len(val_df) == 10
    assert len(train_df) == 80


def test_split_dataset_stratification(tmp_path: Path):
    """Verify that train, val, and test splits preserve class distribution."""
    # 60 class 0, 40 class 1 (40% class 1)
    texts = [f"Unique review text sample index {i} with sufficient words." for i in range(100)]
    labels = [0] * 60 + [1] * 40
    df = pd.DataFrame({"text": texts, "label": labels})

    train_df, val_df, test_df = split_dataset(
        df,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        output_dir=tmp_path,
    )

    # 40% of test (10 rows) is 4
    assert (test_df["label"] == 1).sum() == 4
    # 40% of val (10 rows) is 4
    assert (val_df["label"] == 1).sum() == 4
    # 40% of train (80 rows) is 32
    assert (train_df["label"] == 1).sum() == 32


def test_split_dataset_no_leakage(tmp_path: Path):
    """Verify that there is zero overlap between train, val, and test splits."""
    texts = [f"Unique distinct review text for row {i} with many words." for i in range(60)]
    labels = [0] * 30 + [1] * 30
    df = pd.DataFrame({"text": texts, "label": labels})

    train_df, val_df, test_df = split_dataset(df, output_dir=tmp_path)

    train_set = set(train_df["text"])
    val_set = set(val_df["text"])
    test_set = set(test_df["text"])

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_cli_execution(tmp_path: Path):
    """Verify CLI command executes load, clean, split and outputs CSVs."""
    raw_csv = tmp_path / "raw.csv"
    out_dir = tmp_path / "processed"
    texts = [
        f"This is a detailed product review number {i} with plenty of words to pass the check."
        for i in range(30)
    ]
    labels = ["OR" if i % 2 == 0 else "CG" for i in range(30)]
    pd.DataFrame({"text_": texts, "label": labels}).to_csv(raw_csv, index=False)

    cmd = [
        sys.executable,
        "-m",
        "ml.data",
        "--input",
        str(raw_csv),
        "--output-dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    assert "REVIEWLENS DATA CLEANING REPORT" in result.stdout
    assert "Splits successfully created" in result.stdout
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "val.csv").exists()
    assert (out_dir / "test.csv").exists()

