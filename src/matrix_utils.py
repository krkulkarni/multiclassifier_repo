# src/matrix_utils.py
import numpy as np
import math
import logging
from typing import Union

def symmetric_matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    """Extracts the upper triangle (excluding diagonal) of a symmetric matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square 2D matrix.")
    return matrix[np.triu_indices(matrix.shape[0], k=1)]

def _vector_to_upper_triangle_matrix(vector: np.ndarray) -> np.ndarray:
    """Converts a vector (upper triangle) into an upper triangle matrix."""
    if vector.ndim != 1:
        raise ValueError("Input vector must be 1D.")
    # Solve quadratic equation: n^2 - n - 2*vector_len = 0 to find matrix dimension n
    vector_len = len(vector)
    a = 1
    b = -1
    c = -2 * vector_len
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError(f"Cannot form a square matrix from vector of length {vector_len}")
    n = ( -b + math.sqrt(discriminant) ) / (2*a)
    if not n.is_integer() or n <= 0:
         raise ValueError(f"Cannot form a square matrix from vector of length {vector_len}")
    n = int(n)

    matrix = np.zeros((n, n))
    matrix[np.triu_indices(n, k=1)] = vector
    return matrix

def vector_to_symmetric_matrix(vector: np.ndarray) -> np.ndarray:
    """Converts a vector (upper triangle) into a symmetric matrix."""
    ut_matrix = _vector_to_upper_triangle_matrix(vector)
    return ut_matrix + ut_matrix.T # Add transpose to fill lower triangle

# Alias for backward compatibility if needed in notebook 2
ut_vec_to_symm_mat = vector_to_symmetric_matrix
convert_upper_mat = vector_to_symmetric_matrix # Another alias found in utils.py

# --- Other potentially useful matrix utils ---
def digitize_rdm(rdm_raw: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Digitize an input square matrix into n bins."""
    if rdm_raw.ndim != 2 or rdm_raw.shape[0] != rdm_raw.shape[1]:
        raise ValueError("Input must be a square 2D matrix.")
    # Handle NaNs if necessary
    finite_vals = rdm_raw[np.isfinite(rdm_raw)].flatten()
    if len(finite_vals) == 0:
        logging.warning("Matrix contains no finite values for digitization.")
        return rdm_raw # Or return np.nan matrix

    rdm_bins = np.percentile(finite_vals, np.linspace(0, 100, n_bins + 1)[1:-1]) # Bins between percentiles
    rdm_digitized = np.digitize(rdm_raw, bins=rdm_bins)
    # Force symmetry (might not be ideal depending on use case)
    # rdm_digitized = (rdm_digitized + rdm_digitized.T) / 2
    return rdm_digitized