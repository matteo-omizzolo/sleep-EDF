"""
Hungarian algorithm for label alignment.

Maps unsupervised cluster labels to supervised ground truth labels
to maximize agreement.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Dict


def compute_cost_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Compute cost matrix for Hungarian algorithm.
    
    Cost[i, j] = number of epochs where true label is i and predicted label is j.
    We want to maximize this, so use negative for linear_sum_assignment (minimization).
    
    Args:
        y_true: True labels (n_epochs,)
        y_pred: Predicted labels (n_epochs,)
    
    Returns:
        Cost matrix (n_true_classes, n_pred_classes)
    """
    n_true = len(np.unique(y_true))
    n_pred = len(np.unique(y_pred))
    
    # Build confusion matrix (agreement counts)
    cost = np.zeros((n_true, n_pred))
    
    for i in range(n_true):
        for j in range(n_pred):
            cost[i, j] = np.sum((y_true == i) & (y_pred == j))
    
    # Negate for minimization
    return -cost


def hungarian_alignment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_mapping: bool = True,
    allow_many_to_one: bool = True
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Align predicted cluster labels to true labels using Hungarian algorithm.
    
    Finds optimal one-to-one mapping between predicted and true labels
    that maximizes agreement. If allow_many_to_one=True, handles cases where
    more predicted clusters than true classes by assigning extra clusters
    to their most frequent true label (greedy many-to-one mapping).
    
    Args:
        y_true: True labels (n_epochs,)
        y_pred: Predicted labels (n_epochs,)
        return_mapping: Whether to return mapping dictionary
        allow_many_to_one: If True, allow multiple predicted clusters to map to same true label
    
    Returns:
        y_pred_aligned: Aligned predicted labels (n_epochs,)
        mapping: Dictionary mapping pred_label -> true_label (if return_mapping=True)
    """
    # Get unique labels
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    
    # Compute cost matrix
    cost = compute_cost_matrix(y_true, y_pred)
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Build mapping: pred_label -> true_label
    mapping = {}
    for true_idx, pred_idx in zip(row_ind, col_ind):
        pred_label = pred_labels[pred_idx] if pred_idx < len(pred_labels) else pred_idx
        true_label = true_labels[true_idx] if true_idx < len(true_labels) else true_idx
        mapping[pred_label] = true_label
    
    # Handle unmatched predicted labels (if more predicted than true classes)
    if allow_many_to_one:
        # For each unmapped predicted cluster, assign to most frequent true label
        for pred_label in pred_labels:
            if pred_label not in mapping:
                # Find most common true label for this predicted cluster
                mask = (y_pred == pred_label)
                if np.sum(mask) > 0:
                    true_counts = np.bincount(y_true[mask].astype(int))
                    most_common_true = np.argmax(true_counts)
                    mapping[pred_label] = most_common_true
                else:
                    # Fallback: assign to most frequent overall true label
                    mapping[pred_label] = np.argmax(np.bincount(y_true.astype(int)))
    else:
        # Original behavior: assign to -1 (will be filtered out in metrics)
        for pred_label in pred_labels:
            if pred_label not in mapping:
                mapping[pred_label] = -1
    
    # Apply mapping to get aligned labels
    y_pred_aligned = np.array([mapping.get(label, -1) for label in y_pred])
    
    if return_mapping:
        return y_pred_aligned, mapping
    else:
        return y_pred_aligned


def fit_mapping_on_train_apply_to_test(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Learn label mapping on training set, apply to test set.
    
    This is the correct cross-validation procedure:
    1. Learn mapping on training data
    2. Apply same mapping to test predictions
    
    Args:
        y_train_true: True labels for training set
        y_train_pred: Predicted labels for training set
        y_test_pred: Predicted labels for test set
    
    Returns:
        y_test_aligned: Aligned test predictions
        mapping: Learned mapping dictionary
    """
    # Learn mapping on training data
    _, mapping = hungarian_alignment(y_train_true, y_train_pred, return_mapping=True)
    
    # Apply mapping to test data
    y_test_aligned = np.array([mapping.get(label, -1) for label in y_test_pred])
    
    return y_test_aligned, mapping
