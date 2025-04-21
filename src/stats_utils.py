# src/stats_utils.py
import numpy as np
from typing import Literal, Optional, Tuple
from sklearn.metrics import accuracy_score

def get_permutation_pvalue(
    observed_statistic: float,
    permutation_statistics: np.ndarray,
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
) -> float:
    """
    Calculates the p-value from a permutation distribution.

    Args:
        observed_statistic: The statistic calculated on the original data.
        permutation_statistics: Array of statistics calculated on permuted data.
        alternative: The alternative hypothesis ('two-sided', 'greater', or 'less').

    Returns:
        The calculated p-value.
    """
    n_perms = len(permutation_statistics)
    if n_perms == 0:
        return np.nan

    if alternative == 'greater':
        p_value = (np.sum(permutation_statistics >= observed_statistic) + 1) / (n_perms + 1)
    elif alternative == 'less':
        p_value = (np.sum(permutation_statistics <= observed_statistic) + 1) / (n_perms + 1)
    elif alternative == 'two-sided':
        # Count how many permuted stats are as extreme or more extreme than observed
        abs_observed = np.abs(observed_statistic)
        abs_perms = np.abs(permutation_statistics)
        p_value = (np.sum(abs_perms >= abs_observed) + 1) / (n_perms + 1)
        # An alternative two-sided definition often used:
        # p_greater = (np.sum(permutation_statistics >= observed_statistic) + 1) / (n_perms + 1)
        # p_lesser = (np.sum(permutation_statistics <= observed_statistic) + 1) / (n_perms + 1)
        # p_value = 2 * min(p_greater, p_lesser)
        # p_value = min(p_value, 1.0) # Cap at 1.0
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return p_value

# Add this function to src/stats_utils.py

# Ensure numpy and accuracy_score are imported
import numpy as np
from sklearn.metrics import accuracy_score
# Ensure get_permutation_pvalue is also in this file or imported

def permutation_test_accuracy(
    y_true: np.ndarray,
    y_pred_observed: np.ndarray,
    predict_function: callable, # Function that takes X_test_modified and returns predictions
    X_test_modified: np.ndarray,
    n_permutations: int = 1000,
    random_state: Optional[int] = None
) -> Tuple[float, np.ndarray, float]:
    """
    Performs a permutation test for classifier accuracy.

    Args:
        y_true: True labels for the test set.
        y_pred_observed: Predictions from the classifier on the original test set.
        predict_function: A function (e.g., `classifier.predict`) that takes the
                          modified test data (X_test_modified) and returns predictions.
        X_test_modified: The test data (potentially modified, e.g., zeroed-out features).
        n_permutations: Number of permutations to perform.
        random_state: Seed for the random number generator for reproducibility.

    Returns:
        Tuple containing:
        - observed_accuracy (float): Accuracy on the original data.
        - permuted_accuracies (np.ndarray): Array of accuracies from permuted labels.
        - p_value (float): Permutation p-value (one-sided, greater).
    """
    if n_permutations <= 0:
        return accuracy_score(y_true, y_pred_observed), np.array([]), np.nan

    rng = np.random.default_rng(random_state)
    observed_accuracy = accuracy_score(y_true, y_pred_observed)
    permuted_accuracies = np.zeros(n_permutations)
    y_true_shuffled = y_true.copy() # Copy to avoid modifying original

    for i in range(n_permutations):
        rng.shuffle(y_true_shuffled) # Shuffle true labels
        # Predictions remain the same as they depend on X_test_modified, not y_true
        permuted_accuracies[i] = accuracy_score(y_true_shuffled, y_pred_observed)

    # Calculate p-value (how often permuted accuracy was >= observed accuracy)
    # Use the get_permutation_pvalue function if available and suitable
    p_value = get_permutation_pvalue(observed_accuracy, permuted_accuracies, alternative='greater')
    # Or calculate directly:
    # p_value = (np.sum(permuted_accuracies >= observed_accuracy) + 1) / (n_permutations + 1)

    return observed_accuracy, permuted_accuracies, p_value