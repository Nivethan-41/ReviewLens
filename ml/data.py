"""Dataset ingestion, cleaning, and splitting module for ReviewLens.

Provides functionality to:
1. Load raw review datasets with configurable column mappings and label normalization.
2. Clean text data by removing nulls, stripping whitespace, deduplicating,
   and filtering out overly short reviews.
3. Perform a stratified train/validation/test split avoiding cross-split leakage.
4. CLI interface for executing data pipeline end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import PROCESSED_DATA_DIR, RANDOM_SEED, TEST_SIZE, VAL_SIZE

# Configurable column mapping for raw datasets.
# Users can adapt this module to new datasets by editing this dictionary.
COLUMN_MAPPING: dict[str, Any] = {
    "text_col": "text_",  # Column name for review text (e.g., 'text_', 'text', 'review')
    "label_col": "label",  # Column name for label (e.g., 'label', 'deceptive', 'target')
    "rating_col": "rating",  # Optional column name for star rating (e.g., 'rating', 'overall')
    "category_col": "category",  # Optional column name for product/service category
    "label_map": {
        # Genuine-looking -> 0
        "truthful": 0,
        "genuine": 0,
        "real": 0,
        "true": 0,
        "OR": 0,  # "Original" in Kaggle fake reviews dataset
        "original": 0,
        "0": 0,
        0: 0,
        0.0: 0,
        # Potentially deceptive -> 1
        "deceptive": 1,
        "fake": 1,
        "false": 1,
        "CG": 1,  # "Computer-generated" in Kaggle fake reviews dataset
        "computer_generated": 1,
        "1": 1,
        1: 1,
        1.0: 1,
    },
}


def _normalize_label(val: Any, label_map: dict[Any, int]) -> int:
    """Map a raw label value to 0 (genuine) or 1 (deceptive).

    Args:
        val: Raw label value from dataset.
        label_map: Mapping dictionary.

    Returns:
        Normalized integer label (0 or 1).

    Raises:
        ValueError: If label value cannot be resolved to 0 or 1.
    """
    if val in label_map:
        return label_map[val]

    if isinstance(val, str):
        val_clean = val.strip()
        if val_clean in label_map:
            return label_map[val_clean]
        if val_clean.lower() in label_map:
            return label_map[val_clean.lower()]
        if val_clean.upper() in label_map:
            return label_map[val_clean.upper()]

    try:
        val_int = int(float(val))
        if val_int in label_map:
            return label_map[val_int]
    except (ValueError, TypeError):
        pass

    raise ValueError(
        f"Unrecognized label value '{val}'. Must map to 0 (genuine) or 1 (deceptive)."
    )


def load_raw(
    path: str | Path,
    mapping: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a raw review dataset and normalize column names and labels.

    Normalizes labels to:
    - 0 = genuine-looking
    - 1 = potentially deceptive

    Args:
        path: Path to the raw CSV file.
        mapping: Optional custom column mapping. If None, uses COLUMN_MAPPING.

    Returns:
        Standardized DataFrame with columns 'text', 'label', and optionally 'rating', 'category'.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If required columns ('text_col' or 'label_col') cannot be resolved.
        ValueError: If labels cannot be normalized to 0 or 1.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")

    cfg = mapping if mapping is not None else COLUMN_MAPPING
    df = pd.read_csv(file_path)

    # Resolve text column
    text_col = cfg.get("text_col", "text")
    if text_col not in df.columns:
        # Fallback check for common alternative names if default wasn't found
        fallbacks = ["text", "text_", "review", "review_text", "reviewText", "Review"]
        found = [col for col in fallbacks if col in df.columns]
        if found:
            text_col = found[0]
        else:
            raise KeyError(
                f"Text column '{text_col}' not found in CSV. Available columns: {list(df.columns)}"
            )

    # Resolve label column
    label_col = cfg.get("label_col", "label")
    if label_col not in df.columns:
        fallbacks = ["label", "deceptive", "target", "class", "Label"]
        found = [col for col in fallbacks if col in df.columns]
        if found:
            label_col = found[0]
        else:
            raise KeyError(
                f"Label column '{label_col}' not found in CSV. Available columns: {list(df.columns)}"
            )

    label_map = cfg.get("label_map", COLUMN_MAPPING["label_map"])

    # Standardize column renaming
    rename_dict = {text_col: "text", label_col: "label"}

    rating_col = cfg.get("rating_col", "rating")
    if rating_col in df.columns:
        rename_dict[rating_col] = "rating"
    else:
        rating_fallbacks = ["rating", "stars", "overall", "Rating"]
        found_rating = [col for col in rating_fallbacks if col in df.columns]
        if found_rating:
            rename_dict[found_rating[0]] = "rating"

    category_col = cfg.get("category_col", "category")
    if category_col in df.columns:
        rename_dict[category_col] = "category"
    else:
        category_fallbacks = ["category", "product_category", "Category"]
        found_cat = [col for col in category_fallbacks if col in df.columns]
        if found_cat:
            rename_dict[found_cat[0]] = "category"

    df = df.rename(columns=rename_dict)

    # Map labels to 0 and 1
    df["label"] = df["label"].apply(lambda v: _normalize_label(v, label_map))
    df["label"] = df["label"].astype(int)

    # Keep only standardized columns if present
    desired_cols = ["text", "label"]
    if "rating" in df.columns:
        desired_cols.append("rating")
    if "category" in df.columns:
        desired_cols.append("category")

    return df[desired_cols].copy()


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean the review dataset.

    Performs the following cleaning operations:
    1. Drops null or empty text reviews.
    2. Strips leading and trailing whitespace.
    3. Drops exact duplicate texts (keeping the first occurrence).
    4. Drops reviews shorter than 5 words.

    Args:
        df: Input DataFrame containing at least 'text' and 'label' columns.

    Returns:
        A tuple of (cleaned_df, report_dict) where report_dict contains:
        - rows_before: Total row count before cleaning.
        - rows_after: Total row count after cleaning.
        - duplicates_removed: Number of exact duplicate rows removed.
        - null_or_empty_removed: Number of null/empty text rows removed.
        - short_reviews_removed: Number of reviews with <5 words removed.
        - class_balance: Dict mapping class labels (0, 1) to row counts.
        - class_balance_pct: Dict mapping class labels (0, 1) to proportion of total.
    """
    rows_before = len(df)
    cleaned = df.copy()

    # 1 & 2. Drop nulls, strip whitespace, and drop empty text
    cleaned = cleaned.dropna(subset=["text"])
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned[cleaned["text"] != ""]
    null_or_empty_removed = rows_before - len(cleaned)

    # 3. Drop exact duplicate texts (keep first)
    rows_pre_dup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["text"], keep="first")
    duplicates_removed = rows_pre_dup - len(cleaned)

    # 4. Drop reviews shorter than 5 words
    rows_pre_short = len(cleaned)
    word_counts = cleaned["text"].apply(lambda s: len(s.split()))
    cleaned = cleaned[word_counts >= 5]
    short_reviews_removed = rows_pre_short - len(cleaned)

    cleaned = cleaned.reset_index(drop=True)
    rows_after = len(cleaned)

    # Calculate class balance
    counts = cleaned["label"].value_counts().to_dict()
    class_balance = {0: int(counts.get(0, 0)), 1: int(counts.get(1, 0))}
    total = rows_after
    class_balance_pct = {
        0: round(float(class_balance[0] / total), 4) if total > 0 else 0.0,
        1: round(float(class_balance[1] / total), 4) if total > 0 else 0.0,
    }

    report: Dict[str, Any] = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": duplicates_removed,
        "null_or_empty_removed": null_or_empty_removed,
        "short_reviews_removed": short_reviews_removed,
        "class_balance": class_balance,
        "class_balance_pct": class_balance_pct,
    }

    return cleaned, report


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_SEED,
    output_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a stratified train/validation/test split and save CSVs.

    Ensures no duplicate texts leak across splits. Saves:
    - train.csv
    - val.csv
    - test.csv
    to the designated output directory (defaults to data/processed/).

    Args:
        df: Cleaned DataFrame with 'text' and 'label' columns.
        test_size: Proportion of dataset allocated to the test set.
        val_size: Proportion of dataset allocated to the validation set.
        random_state: Random seed for reproducibility.
        output_dir: Output directory path. Defaults to PROCESSED_DATA_DIR.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If splits leak duplicate reviews or stratification fails.
    """
    # Defensive check: ensure no exact duplicate texts exist in input
    deduped = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    # First split: separate test set from train + validation
    train_val_df, test_df = train_test_split(
        deduped,
        test_size=test_size,
        random_state=random_state,
        stratify=deduped["label"],
    )

    # Second split: separate validation set from train
    val_relative_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_ratio,
        random_state=random_state,
        stratify=train_val_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Assert no duplicate text leakage across splits
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    if not train_texts.isdisjoint(val_texts):
        overlap = train_texts.intersection(val_texts)
        raise ValueError(f"Data leakage detected! {len(overlap)} text(s) overlap between train and val.")
    if not train_texts.isdisjoint(test_texts):
        overlap = train_texts.intersection(test_texts)
        raise ValueError(f"Data leakage detected! {len(overlap)} text(s) overlap between train and test.")
    if not val_texts.isdisjoint(test_texts):
        overlap = val_texts.intersection(test_texts)
        raise ValueError(f"Data leakage detected! {len(overlap)} text(s) overlap between val and test.")

    # Save to disk
    target_dir = Path(output_dir) if output_dir is not None else PROCESSED_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(target_dir / "train.csv", index=False)
    val_df.to_csv(target_dir / "val.csv", index=False)
    test_df.to_csv(target_dir / "test.csv", index=False)

    return train_df, val_df, test_df


def main() -> None:
    """CLI entry point for running data pipeline."""
    parser = argparse.ArgumentParser(
        description="ReviewLens Data Pipeline: Ingest raw reviews, clean, and split."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to raw CSV dataset (e.g. data/raw/fake_reviews.csv)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Optional output directory for processed CSVs (defaults to data/processed)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DATA_DIR

    print(f"Loading raw dataset from '{input_path}'...")
    raw_df = load_raw(input_path)

    print(f"Loaded {len(raw_df)} rows. Cleaning dataset...")
    cleaned_df, report = clean_dataset(raw_df)

    print("\n" + "=" * 55)
    print("REVIEWLENS DATA CLEANING REPORT")
    print("=" * 55)
    print(f"Total rows before cleaning:   {report['rows_before']}")
    print(f"Null or empty text dropped:   {report['null_or_empty_removed']}")
    print(f"Duplicate reviews removed:    {report['duplicates_removed']}")
    print(f"Short reviews (<5 words):     {report['short_reviews_removed']}")
    print(f"Total rows after cleaning:    {report['rows_after']}")
    print("Class balance (0 = Genuine-looking, 1 = Potentially deceptive):")
    for cls, cnt in report["class_balance"].items():
        pct = report["class_balance_pct"].get(cls, 0.0) * 100
        cls_name = "Genuine (0)" if cls == 0 else "Deceptive (1)"
        print(f"  - {cls_name}: {cnt} reviews ({pct:.1f}%)")
    print("=" * 55)

    print(f"\nSplitting dataset into train/val/test (stratified)...")
    train_df, val_df, test_df = split_dataset(cleaned_df, output_dir=output_dir)

    total_clean = len(cleaned_df)
    print(f"\nSplits successfully created and saved to '{output_dir}':")
    print(f"  - train.csv: {len(train_df)} rows ({len(train_df)/total_clean*100:.1f}%)")
    print(f"  - val.csv:   {len(val_df)} rows ({len(val_df)/total_clean*100:.1f}%)")
    print(f"  - test.csv:  {len(test_df)} rows ({len(test_df)/total_clean*100:.1f}%)")
    print("=" * 55)
    print("Data processing complete!")


if __name__ == "__main__":
    main()
