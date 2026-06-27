from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def dense_float32_matrix(matrix, *, dataset_label: str) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{dataset_label} must be 2D.")
    if not np.isfinite(values).all():
        raise ValueError(f"{dataset_label} must contain only finite values.")
    return values


def unique_in_order(values) -> list:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def normalise(values: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        raise ValueError(f"{label} must have positive total mass.")
    return values / total


def probabilities_to_logits(probabilities: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    return np.log(np.asarray(probabilities, dtype=np.float64) + eps).astype(np.float32)


def softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def centre_rows(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=1, keepdims=True)


def l2_normalise_rows(values: np.ndarray, *, eps: float = 1e-8, allow_zero_rows: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    zero_rows = norms[:, 0] <= eps
    if np.any(zero_rows) and not allow_zero_rows:
        raise ValueError("l2_normalise_rows(...): all rows must have positive norm.")
    output = np.zeros_like(values, dtype=np.float64)
    nonzero_rows = ~zero_rows
    output[nonzero_rows] = values[nonzero_rows] / norms[nonzero_rows]
    return output


def validate_labels(name: str, values: np.ndarray) -> None:
    if pd.isna(values).any():
        raise ValueError(f"{name} contains missing values.")
