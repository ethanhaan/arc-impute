from __future__ import annotations

from anndata import AnnData
import numpy as np
import pandas as pd

from arc_impute._utils import dense_float32_matrix


def validate_assignment(
    assignment,
    *,
    n_reference_cells: int,
    n_spatial_spots: int,
) -> np.ndarray:
    assignment = np.asarray(assignment, dtype=np.float32)
    if assignment.shape != (n_reference_cells, n_spatial_spots):
        raise ValueError("ARC-Impute assignment must have shape reference cells x spatial spots.")
    if not np.isfinite(assignment).all():
        raise ValueError("ARC-Impute assignment must contain only finite values.")
    if (assignment < 0).any():
        raise ValueError("ARC-Impute assignment must be non-negative.")
    return assignment


def impute_expression(
    *,
    assignment: np.ndarray,
    curated_reference: AnnData,
    spot_ids: list[str],
    genes: list[str],
) -> pd.DataFrame:
    genes = list(genes)
    if len(genes) == 0:
        raise ValueError("ArcImpute.impute(...): genes must be non-empty.")
    if len(set(genes)) != len(genes):
        raise ValueError("ArcImpute.impute(...): genes must be unique.")
    missing = [gene for gene in genes if gene not in curated_reference.var_names]
    if missing:
        raise ValueError(f"ArcImpute.impute(...): curated reference is missing gene {missing[0]!r}.")

    reference_expression = dense_float32_matrix(
        curated_reference[:, genes].X,
        dataset_label="curated_reference.X",
    )
    predicted = assignment.T @ reference_expression
    return pd.DataFrame(
        predicted,
        index=pd.Index(spot_ids, name="spot"),
        columns=pd.Index(genes, name="gene"),
    )


def deconvolve_assignment(
    *,
    assignment: np.ndarray,
    cell_types: np.ndarray,
    spot_ids: list[str],
) -> pd.DataFrame:
    labels = pd.Index(sorted(pd.unique(pd.Series(cell_types).astype(str)).tolist()), name="cell_type")
    one_hot = np.zeros((len(cell_types), len(labels)), dtype=np.float32)
    label_to_index = {label: index for index, label in enumerate(labels.tolist())}
    for row_index, label in enumerate(pd.Series(cell_types).astype(str).tolist()):
        one_hot[row_index, label_to_index[label]] = 1.0

    composition = assignment.T @ one_hot
    row_sums = composition.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("ArcImpute.deconvolve(...): every spot must receive positive assignment mass.")
    composition = composition / row_sums[:, None]
    return pd.DataFrame(
        composition,
        index=pd.Index(spot_ids, name="spot"),
        columns=labels,
    )
