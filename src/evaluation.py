"""
Evaluation Module

Contains methods to evaluate the performance of Automated ICD Coding models.
"""

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_metrics(y_true, y_pred, y_pred_prob=None, k_values=[5, 8, 15]):
    """
    Calculates primary and secondary evaluation metrics.
    
    Args:
        y_true: True binary labels (sparse or dense array).
        y_pred: Predicted binary labels (sparse or dense array).
        y_pred_prob: Predicted probabilities / decision function for Precision@K calculation (optional).
        k_values: List of K values for Precision@K.

    Returns:
        metrics: Dictionary containing Micro and Macro metrics, plus optionally P@K.
    """
    metrics = {
        'micro_f1': f1_score(y_true, y_pred, average='micro'),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'micro_precision': precision_score(y_true, y_pred, average='micro'),
        'micro_recall': recall_score(y_true, y_pred, average='micro'),
    }
    
    # Calculate Precision@K if probabilities/decision functions are available
    if y_pred_prob is not None:
        for k in k_values:
            p_at_k = precision_at_k(y_true, y_pred_prob, k)
            metrics[f'P@{k}'] = p_at_k

    return metrics

def precision_at_k(y_true, y_prob, k=5):
    """
    Computes Precision@K.
    """
    # Convert true labels to dense numpy array if sparse
    if hasattr(y_true, 'toarray'):
        y_true = y_true.toarray()
    if hasattr(y_prob, 'toarray'):
        y_prob = y_prob.toarray()
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    precisions = []
    for i in range(y_true.shape[0]):
        # Get top k indices
        top_k_indices = np.argsort(y_prob[i])[::-1][:k]
        
        # Calculate precision: true positives within top k / k
        tp = np.sum(y_true[i, top_k_indices])
        precisions.append(tp / k)
        
    return np.mean(precisions)

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
    
    # We mainly plot Micro-F1 and Macro-F1 unless only one is present
    plot_cols = [col for col in ['micro_f1', 'macro_f1'] if col in df.columns]
    
    if not plot_cols:
        plot_cols = df.columns.tolist()

    df[plot_cols].plot(kind='bar', figsize=(10, 5), title='Model Comparison (F1 Scores)')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
        
    plt.show()
