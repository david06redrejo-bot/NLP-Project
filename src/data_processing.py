"""
Data Processing Module for Automated ICD Coding

Preprocessing pipeline for Spanish/Catalan clinical text and
dataset preparation for multi-label classification.
"""

import re
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


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
    text = re.sub(r'<[^>]+>', ' ', text)

    # Normalize unicode (NFC keeps ñ, accented vowels intact)
    text = unicodedata.normalize('NFC', text)

    # Keep only letters (including Spanish/Catalan), digits, and whitespace
    text = re.sub(r'[^a-záéíóúüñçà-ú0-9\s]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_texts(texts: list) -> list:
    """Apply clean_text to a list of strings."""
    return [clean_text(t) for t in texts]


# ── Dataset Preparation ──────────────────────────────────────────────────────

def prepare_multilabel_dataset(
    df: pd.DataFrame,
    literal_col: str = 'Literal',
    code_col: str = 'Code',
    min_code_freq: int = 10,
):
    """
    Transform the raw codification CSV into multi-label format.

    Steps:
        1. Count code frequencies and keep only codes with >= min_code_freq
        2. Group by literal: each unique literal → list of its codes
        3. Drop literals that have no remaining codes after filtering

    Args:
        df:            DataFrame with at least [literal_col, code_col]
        literal_col:   Column name for clinical text
        code_col:      Column name for ICD-10 codes
        min_code_freq: Minimum number of occurrences for a code to be kept

    Returns:
        X:             list of literal strings
        y:             list of lists of ICD code strings
        kept_codes:    sorted list of codes that passed the frequency filter
    """
    # Filter to frequent codes
    code_counts = df[code_col].value_counts()
    frequent_codes = set(code_counts[code_counts >= min_code_freq].index)

    print(f"Total unique codes: {len(code_counts):,}")
    print(f"Codes with >= {min_code_freq} occurrences: {len(frequent_codes):,}")

    # Group by literal
    grouped = df.groupby(literal_col)[code_col].apply(list).reset_index()
    grouped.columns = ['Literal', 'Codes']

    # Keep only codes that passed the filter
    grouped['Codes'] = grouped['Codes'].apply(
        lambda cs: sorted(set(c for c in cs if c in frequent_codes))
    )

    # Drop rows with no remaining codes
    grouped = grouped[grouped['Codes'].str.len() > 0].reset_index(drop=True)

    X = grouped['Literal'].tolist()
    y = grouped['Codes'].tolist()
    kept_codes = sorted(frequent_codes)

    print(f"Literals retained: {len(X):,}")
    print(f"Avg codes per literal: {sum(len(c) for c in y) / len(y):.2f}")

    return X, y, kept_codes


def split_and_binarize(
    X: list,
    y: list,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split into train/val and binarize labels.

    Args:
        X:            list of literal strings
        y:            list of lists of ICD code strings
        test_size:    fraction for validation set
        random_state: reproducibility seed

    Returns:
        X_train, X_val:         string lists
        y_train_bin, y_val_bin: sparse binary matrices
        mlb:                    fitted MultiLabelBinarizer
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    y_val_bin = mlb.transform(y_val)

    print(f"Train: {len(X_train):,} samples | Val: {len(X_val):,} samples")
    print(f"Label matrix shape: {y_train_bin.shape}")

    return X_train, X_val, y_train_bin, y_val_bin, mlb
