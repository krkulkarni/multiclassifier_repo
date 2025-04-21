# src/modeling_utils.py
import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score
from typing import Tuple

def sigmoid(x: float) -> float:
    """Calculates the sigmoid function, handling potential overflow."""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        # Return 0 or 1 depending on the sign of x for large inputs
        return 0.0 if x < 0 else 1.0


def get_distances(X: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    """Calculates the linear decision function values."""
    if X.shape[1] != weights.shape[0]:
        raise ValueError(f"Feature mismatch: X has {X.shape[1]} features, weights has {weights.shape[0]}.")
    return X @ weights + intercept # Use matrix multiplication for efficiency

def get_probabilities(X: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    """Calculates predicted probabilities using the sigmoid function."""
    distances = get_distances(X, weights, intercept)
    # Apply sigmoid element-wise using vectorize for potential speedup on arrays
    sigmoid_vec = np.vectorize(sigmoid)
    return sigmoid_vec(distances)

def find_optimal_clf_threshold(
    train_dists: np.ndarray,
    y_train: np.ndarray
) -> pd.DataFrame:
    """
    Finds the classification threshold on distances that maximizes accuracy on training data.

    Args:
        train_dists: Decision function values (distances) for the training set.
        y_train: True labels for the training set.

    Returns:
        Pandas DataFrame containing ['percentile', 'threshold', 'accuracy'] sorted by accuracy descending.
    """
    accuracies = []
    # Iterate through possible thresholds based on sorted distances
    # Adding small epsilon to boundaries to handle edge cases
    unique_sorted_dists = np.sort(np.unique(train_dists))
    thresholds = np.concatenate(([unique_sorted_dists[0] - 1e-6],
                                 (unique_sorted_dists[:-1] + unique_sorted_dists[1:]) / 2, # Midpoints
                                 [unique_sorted_dists[-1] + 1e-6])) # Boundaries

    for threshold in thresholds:
        preds = (train_dists > threshold).astype(int)
        acc = accuracy_score(y_train, preds)
        # Calculate percentile corresponding to this threshold (approximate)
        percentile = (np.sum(train_dists < threshold) / len(train_dists)) * 100
        accuracies.append({'percentile': percentile, 'threshold': threshold, 'accuracy': acc})

    # Also check threshold at 0 explicitly if not already covered
    if 0 not in thresholds:
         preds_zero = (train_dists > 0).astype(int)
         acc_zero = accuracy_score(y_train, preds_zero)
         percentile_zero = (np.sum(train_dists < 0) / len(train_dists)) * 100
         accuracies.append({'percentile': percentile_zero, 'threshold': 0.0, 'accuracy': acc_zero})


    df = pd.DataFrame(accuracies)
    df = df.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
    return df