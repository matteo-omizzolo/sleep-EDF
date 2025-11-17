"""
Evaluation metrics for sleep staging.

Includes:
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Macro-averaged F1 score
- Predictive log-likelihood
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from typing import Optional


def compute_ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index.
    
    Measures clustering agreement between true and predicted labels.
    Corrected for chance (ARI = 1 for perfect agreement, 0 for random).
    
    Args:
        y_true: True labels (n_epochs,)
        y_pred: Predicted labels (n_epochs,)
    
    Returns:
        ARI score
    """
    return adjusted_rand_score(y_true, y_pred)


def compute_nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information.
    
    Measures information-theoretic similarity between clusterings.
    NMI = 1 for perfect agreement, 0 for independent clusterings.
    
    Args:
        y_true: True labels (n_epochs,)
        y_pred: Predicted labels (n_epochs,)
    
    Returns:
        NMI score
    """
    return normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')


def compute_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None
) -> float:
    """
    Compute macro-averaged F1 score.
    
    Average F1 across all classes (unweighted).
    Requires labels to be aligned (e.g., via Hungarian algorithm).
    
    Args:
        y_true: True labels (n_epochs,)
        y_pred: Predicted labels (n_epochs,)
        labels: Optional list of label values to include
    
    Returns:
        Macro F1 score
    """
    return f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)


def predictive_log_likelihood(
    model,
    X: np.ndarray,
    subject_idx: int = 0
) -> float:
    """
    Compute predictive log-likelihood of test sequence.
    
    Args:
        model: Fitted HMM model with log_likelihood method
        X: Test feature matrix (n_epochs, n_features)
        subject_idx: Subject index (for hierarchical models)
    
    Returns:
        Log-likelihood
    """
    return model.log_likelihood(X, subject_idx=subject_idx)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None
) -> dict:
    """
    Compute precision, recall, F1 for each class.
    
    Args:
        y_true: True labels (n_epochs,)
        y_pred: Predicted labels (n_epochs,)
        class_names: Optional list of class names
    
    Returns:
        Dictionary with per-class metrics
    """
    from sklearn.metrics import classification_report
    
    if class_names is None:
        class_names = [f"Class {i}" for i in np.unique(y_true)]
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    return report


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels (n_epochs,)
        y_pred: Predicted labels (n_epochs,)
        normalize: Normalization mode ('true', 'pred', 'all', or None)
    
    Returns:
        Confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    
    return confusion_matrix(y_true, y_pred, normalize=normalize)
