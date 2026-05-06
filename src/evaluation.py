"""
Evaluation Module

Contains methods to evaluate the performance of ICD category classification
models and to generate the required submission files.
"""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import pandas as pd
import matplotlib.pyplot as plt


def calculate_category_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for single-label ICD category classification.

    The official metric is strict accuracy (exact match on the first digit),
    but we also report weighted and macro F1 for a fuller picture.

    Args:
        y_true: list/array of true category strings.
        y_pred: list/array of predicted category strings.

    Returns:
        metrics: Dictionary of metric name → score.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }
    return metrics


def print_classification_report(y_true, y_pred):
    """Print sklearn's per-class classification report."""
    print(classification_report(y_true, y_pred, zero_division=0))


def print_comparison_table(results_dict):
    """
    Prints a comparison table of evaluation metrics.

    Args:
        results_dict: Dictionary where keys are model names and values are metric dictionaries.
    """
    df = pd.DataFrame(results_dict).T
    print("\n--- Model Comparison ---")
    print(df.to_string())
    return df


def plot_comparison(results_dict, save_path=None):
    """
    Plots a bar chart comparing models across metrics and optionally saves it.

    Args:
        results_dict: Dictionary where keys are model names and values are metric dictionaries.
        save_path: Optional path to save the generated plot.
    """
    df = pd.DataFrame(results_dict).T

    plot_cols = [
        col for col in ["accuracy", "weighted_f1", "macro_f1"] if col in df.columns
    ]
    if not plot_cols:
        plot_cols = df.columns.tolist()

    df[plot_cols].plot(kind="bar", figsize=(10, 5), title="Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")

    plt.show()


def generate_submission(
    leaderboard_df,
    y_pred_categories,
    output_path: str,
    id_col: str = "id",
    literal_col: str = "Literal",
):
    """
    Build the required submission CSV for the ICD category challenge.

    Required columns (in order):
        id         – original id for each row (FIRST COLUMN)
        Literal    – original literal value
        y_category – predicted ICD category (first digit); no empty values

    Args:
        leaderboard_df:    DataFrame with [id_col, literal_col].
        y_pred_categories: list/array of predicted category strings (one per row).
        output_path:       Where to write the CSV.
        id_col:            Column name for the sample identifier.
        literal_col:       Column name for the literal text.

    Returns:
        submission_df: The generated DataFrame.
    """
    submission_df = pd.DataFrame(
        {
            "id": leaderboard_df[id_col].values,
            "Literal": leaderboard_df[literal_col].values,
            "y_category": y_pred_categories,
        }
    )

    # Ensure no empty predictions — fill with "null" as required
    submission_df["y_category"] = submission_df["y_category"].fillna("null")
    submission_df["y_category"] = submission_df["y_category"].replace("", "null")

    # Ensure id is the first column
    submission_df = submission_df[["id", "Literal", "y_category"]]

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    print(f"  Total rows: {len(submission_df):,}")
    print(f"  Unique categories predicted: {submission_df['y_category'].nunique()}")
    print(f"  Any empty values: {(submission_df['y_category'] == 'null').sum()}")
    return submission_df
