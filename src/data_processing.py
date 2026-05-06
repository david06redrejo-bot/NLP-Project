"""
Data Processing Module for Automated ICD Coding

Preprocessing pipeline for Spanish/Catalan clinical text and
dataset preparation for ICD category classification.
"""

import re
import unicodedata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ── Text Cleaning ─────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """
    Clean a clinical literal for feature extraction.

    Steps:
        1. Convert to lowercase
        2. Strip HTML tags (some training literals contain raw HTML)
        3. Normalize unicode accents (keep ñ, á, é, í, ó, ú, ü)
        4. Remove non-alphanumeric characters (preserve Spanish/Catalan chars)
        5. Collapse multiple whitespace into single space
        6. Strip leading/trailing whitespace

    Args:
        text: raw clinical literal string

    Returns:
        Cleaned string ready for vectorization.
    """
    text = str(text).lower().strip()

    # Remove HTML tags like <font>, </font>, etc.
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize unicode (NFC keeps ñ, accented vowels intact)
    text = unicodedata.normalize("NFC", text)

    # Keep only letters (including Spanish/Catalan), digits, and whitespace
    text = re.sub(r"[^a-záéíóúüñçà-ú0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_text(text: str) -> str:
    """
    Normalise a clinical literal following the project proposal pipeline.

    This is the accent-stripping variant used in the TF-IDF + SVM baseline.
    Steps:
        1. Lowercase
        2. Strip HTML tags
        3. Strip ALL diacritical marks / accents (á→a, ñ→n, etc.)
        4. Remove non-alphanumeric characters
        5. Collapse whitespace

    Args:
        text: raw clinical literal string

    Returns:
        Fully normalised string with no accents.
    """
    text = str(text).lower().strip()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # NFD decomposition splits base char + combining diacritic
    text = unicodedata.normalize("NFD", text)
    # Remove combining diacritical marks (category 'Mn')
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # Keep only ASCII letters, digits, whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_texts(texts: list) -> list:
    """Apply clean_text to a list of strings."""
    return [clean_text(t) for t in texts]


def normalize_texts(texts) -> list:
    """Apply normalize_text (accent-stripping) to a list / Series of strings."""
    return [normalize_text(t) for t in texts]


# ── Category Extraction ──────────────────────────────────────────────────────


def extract_category(code: str) -> str:
    """
    Extract the ICD-10 category from a code (its first character, uppercased).

    Examples:
        'J9809' → 'J'
        '07CP0ZZ' → '0'
        'N801' → 'N'

    Args:
        code: Full ICD-10 code string.

    Returns:
        Single-character category string.
    """
    return str(code)[0].upper()


# ── Dataset Preparation ──────────────────────────────────────────────────────


def prepare_category_dataset(
    df: pd.DataFrame,
    literal_col: str = "Literal",
    code_col: str = "Code",
):
    """
    Prepare a single-label dataset for ICD category (first digit) classification.

    For literals that appear with multiple different categories, the most
    frequent category is kept (majority vote).

    Args:
        df:           DataFrame with at least [literal_col, code_col].
        literal_col:  Column name for clinical text.
        code_col:     Column name for ICD-10 codes.

    Returns:
        result_df:    DataFrame with columns [Literal, y_category].
    """
    # Extract category (first character of each code)
    df = df.copy()
    df["y_category"] = df[code_col].apply(extract_category)

    unique_cats = sorted(df["y_category"].unique())
    print(f"Total rows: {len(df):,}")
    print(f"Unique categories: {len(unique_cats)}  →  {unique_cats}")

    # For each literal, take the most frequent category (majority vote)
    grouped = (
        df.groupby(literal_col)["y_category"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    grouped.columns = ["Literal", "y_category"]

    print(f"Unique literals: {len(grouped):,}")
    print(f"Category distribution:")
    print(grouped["y_category"].value_counts().sort_index().to_string())

    return grouped


def split_category_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split a category dataset into train / validation.

    Args:
        df:            DataFrame with [Literal, y_category].
        test_size:     Fraction for validation set.
        random_state:  Reproducibility seed.

    Returns:
        X_train, X_val:   lists of literal strings.
        y_train, y_val:   lists of category strings.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        df["Literal"].tolist(),
        df["y_category"].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=df["y_category"],
    )

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")
    return X_train, X_val, y_train, y_val
